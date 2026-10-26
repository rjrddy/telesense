"""Exercise deployment middleware without loading optional ML dependencies.

Run: python -m unittest discover -s tests -p 'test_demo_middleware.py'
"""
import asyncio
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from fastapi import APIRouter, Request, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


class DemoMiddlewareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Substitute only the heavyweight routes; middleware remains real.
        routes = types.ModuleType('app.backend.api.routes')
        routes.router = APIRouter()

        @routes.router.get('/status/test')
        async def status():
            return {'status': 'ok'}

        @routes.router.get('/audio/test/audio.wav')
        async def audio():
            return {'audio': True}

        @routes.router.post('/upload')
        async def upload(request: Request):
            return {'bytes': len(await request.body())}

        @routes.router.websocket('/ws/test')
        async def websocket(ws: WebSocket):
            await ws.accept()
            await ws.send_json({'status': 'ok'})
            await ws.close()

        spec = importlib.util.spec_from_file_location(
            'app.backend.demo_main_test', Path(__file__).resolve().parents[1] / 'app/backend/main.py'
        )
        cls.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'app.backend.api.routes': routes}), patch.dict(os.environ, {
            'TELESENSE_DEMO_KEY': 'test', 'TELESENSE_MAX_UPLOAD_MB': '1',
            'CORS_ORIGINS': 'http://localhost:5173',
        }):
            spec.loader.exec_module(cls.module)
        cls.client = TestClient(cls.module.app)

    def test_http_auth_and_cors_errors(self):
        self.assertEqual(self.client.get('/health').status_code, 200)
        response = self.client.get('/api/status/test', headers={'Origin': 'http://localhost:5173'})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers['access-control-allow-origin'], 'http://localhost:5173')
        self.assertEqual(self.client.get('/api/status/test?key=test').status_code, 401)
        self.assertEqual(self.client.get('/api/status/test', headers={'X-Telesense-Key': 'test'}).status_code, 200)
        self.assertEqual(self.client.get('/api/status/test', headers={'X-Telesense-Key': 'wrong'}).status_code, 401)

    def test_preflight(self):
        response = self.client.options('/api/upload', headers={
            'Origin': 'http://localhost:5173', 'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'X-Telesense-Key',
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['access-control-allow-origin'], 'http://localhost:5173')

    def test_media_and_websocket_auth(self):
        self.assertEqual(self.client.get('/api/audio/test/audio.wav').status_code, 401)
        self.assertEqual(self.client.get('/api/audio/test/audio.wav?key=test').status_code, 200)
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect('/api/ws/test'):
                pass
        with self.client.websocket_connect('/api/ws/test?key=test') as ws:
            self.assertEqual(ws.receive_json(), {'status': 'ok'})

    def test_upload_declared_and_streamed_limits(self):
        headers = {'X-Telesense-Key': 'test'}
        self.assertEqual(self.client.post('/api/upload', content=b'abc', headers=headers).json(), {'bytes': 3})
        self.assertEqual(self.client.post('/api/upload', content=b'x' * (1024 * 1024 + 1), headers=headers).status_code, 413)
        self.assertEqual(self.client.post('/api/upload', content=iter([b'abc', b'def']), headers=headers).json(), {'bytes': 6})
        self.assertEqual(self.client.post('/api/upload', content=iter([b'x' * 600000] * 2), headers=headers).status_code, 413)

    def test_split_asgi_stream_without_length(self):
        async def run():
            messages = iter([
                {'type': 'http.request', 'body': b'x' * 600000, 'more_body': True},
                {'type': 'http.request', 'body': b'x' * 600000, 'more_body': False},
            ])
            sent = []
            called = False

            async def downstream(scope, receive, send):
                nonlocal called
                called = True

            async def receive():
                return next(messages)

            async def send(message):
                sent.append(message)

            await self.module.SizeLimitMiddleware(downstream)(
                {'type': 'http', 'method': 'POST', 'headers': []}, receive, send
            )
            self.assertFalse(called)
            self.assertEqual(sent[0]['status'], 413)
        asyncio.run(run())
