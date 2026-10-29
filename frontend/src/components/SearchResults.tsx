import type { ReviewHit } from "../api/search";

interface SearchResultsProps {
  hits: ReviewHit[];
}

export function SearchResults({ hits }: SearchResultsProps) {
  if (hits.length === 0) {
    return <p className="status">No matching reviews found.</p>;
  }

  return (
    <ol className="results" aria-label="Hotel search results">
      {hits.map((hit, index) => (
        <li className="result" key={`${hit.hotel_name}-${index}`}>
          <div className="result-heading">
            <h2>{hit.hotel_name}</h2>
            <span>Similarity {hit.score.toFixed(2)}</span>
          </div>
          <p>{hit.review}</p>
        </li>
      ))}
    </ol>
  );
}
