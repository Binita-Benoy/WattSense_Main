/** Notices from the upload checker (old data, summed readings, filled gaps...). */
export default function Warnings({ items }) {
  if (!items || items.length === 0) return null;
  return (
    <div className="info-note">
      <strong>Please note</strong>
      <ul>
        {items.map((w) => (
          <li key={w}>{w}</li>
        ))}
      </ul>
    </div>
  );
}