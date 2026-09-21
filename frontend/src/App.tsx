import { FormEvent, useState } from "react";
import { useMutation } from "@tanstack/react-query";

import { searchHotels } from "./api/search";
import { SearchResults } from "./components/SearchResults";

export default function App() {
  const [query, setQuery] = useState("");
  const search = useMutation({ mutationFn: searchHotels });

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalizedQuery = query.trim();
    if (normalizedQuery) search.mutate(normalizedQuery);
  }

  return (
    <main>
      <header>
        <p className="eyebrow">PARIS, FRANCE</p>
        <h1>Find a hotel that feels right.</h1>
        <p className="intro">
          Describe the stay you want. BonNuit searches real guest reviews for the closest matches.
        </p>
      </header>

      <form className="search" onSubmit={handleSubmit}>
        <label htmlFor="hotel-query">What are you looking for?</label>
        <div className="search-controls">
          <input
            id="hotel-query"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="A quiet hotel near the Louvre"
            autoComplete="off"
          />
          <button disabled={search.isPending || !query.trim()} type="submit">
            {search.isPending ? "Searching…" : "Search"}
          </button>
        </div>
      </form>

      {search.isError && <p className="status error">{search.error.message}</p>}
      {search.data && <SearchResults hits={search.data.hits} />}
    </main>
  );
}
