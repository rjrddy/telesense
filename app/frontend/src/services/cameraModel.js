// Lazy-loaded only after the visitor explicitly starts camera evaluation.
// Model files are served from this site; camera frames never leave the browser.
let loading;
export function loadCameraModel() {
  if (!loading)
    loading = (async () => {
      const faceapi = await import("@vladmandic/face-api/dist/face-api.esm.js");
      try {
        if (!(await faceapi.tf.setBackend("webgl")))
          throw Error("WebGL unavailable");
        await faceapi.tf.ready();
      } catch {
        await faceapi.tf.setBackend("cpu");
        await faceapi.tf.ready();
      }
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri("/models/face-expression"),
        faceapi.nets.faceExpressionNet.loadFromUri("/models/face-expression"),
      ]);
      return faceapi;
    })().catch((error) => {
      loading = null;
      throw error;
    });
  return loading;
}
