import { useEffect, useRef } from "react";

interface Props {
  stream: MediaStream | null;
  label: string;
  muted?: boolean;
}

export function VideoTile({ stream, label, muted }: Props) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    if (ref.current && stream) ref.current.srcObject = stream;
  }, [stream]);
  return (
    <div className="video-tile">
      <video ref={ref} autoPlay playsInline muted={muted} />
      <div className="label">{label}</div>
    </div>
  );
}
