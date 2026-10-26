const paths = {
  grid: "M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z",
  wave: "M3 10v4 M7 6v12 M11 3v18 M15 7v10 M19 5v14 M23 10v4",
  upload: "M12 16V3 M7 8l5-5 5 5 M4 15v5h16v-5",
  arrow: "M5 12h14 M13 6l6 6-6 6",
  back: "M19 12H5 M11 6l-6 6 6 6",
  play: "M8 4l12 8-12 8z",
  pause: "M8 4v16 M16 4v16",
  file: "M14 2H4v20h16V8z M14 2v6h6 M8 13h8 M8 17h5",
  shield: "M12 2l9 4v6c0 5-9 10-9 10S3 17 3 12V6z M8 12l3 3 5-6",
  info: "M12 11v6 M12 7v.1 M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0",
  clock: "M12 6v6l4 2 M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0",
  download: "M12 3v13 M7 11l5 5 5-5 M4 17v4h16v-4",
  chevron: "M9 5l7 7-7 7",
  check: "M5 12l4 4L19 6",
  search: "M21 21l-6-6 M17 10a7 7 0 1 1-14 0 7 7 0 0 1 14 0",
  video: "M3 5h12v14H3z M15 9l6-4v14l-6-4",
  text: "M4 5h16 M12 5v15 M8 20h8",
  close: "M6 6l12 12 M18 6L6 18",
};
export default function Icon({ name, size = 20, ...props }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      <path d={paths[name] || paths.wave} />
    </svg>
  );
}
