import { useEffect, useRef, useState } from "react";

export interface ChatMessage {
  from: string;
  text: string;
}

interface Props {
  messages: ChatMessage[];
  onSend: (text: string) => void;
  me: string;
}

export function Chat({ messages, onSend, me }: Props) {
  const [draft, setDraft] = useState("");
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [messages.length]);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const text = draft.trim();
    if (!text) return;
    onSend(text);
    setDraft("");
  };

  return (
    <>
      <div className="chat-log" ref={logRef}>
        {messages.map((m, i) => (
          <div key={i} className="chat-msg">
            <span className="from">{m.from === me ? "you" : m.from.slice(0, 6)}:</span>
            {m.text}
          </div>
        ))}
      </div>
      <form className="chat-input" onSubmit={submit}>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Type a message…"
        />
        <button type="submit">Send</button>
      </form>
    </>
  );
}
