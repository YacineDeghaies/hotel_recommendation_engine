export interface ReviewHit {
  hotel_name: string;
  review: string;
  score: number;
  locality: string | null;
}

export interface SearchResponse {
  query: string;
  hits: ReviewHit[];
}

export async function searchHotels(query: string): Promise<SearchResponse> {
  const response = await fetch("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, k: 5, locality: "Paris" }),
  });

  if (!response.ok) {
    throw new Error("Search is unavailable. Please try again.");
  }

  return response.json() as Promise<SearchResponse>;
}
