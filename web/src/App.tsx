import { FormEvent, useEffect, useMemo, useState } from "react";
import { fetchArticle, searchHistory } from "./api";
import MapView from "./MapView";
import type { ArticleCard, ArticleDetail, SearchResponse } from "./types";

type MainTab = "map" | "stories" | "timeline";
type RailTab = "mapped" | "connected";

function ArticleRow({
  article,
  onSelect,
}: {
  article: ArticleCard;
  onSelect: (article: ArticleCard) => void;
}) {
  return (
    <button className="article-row" onClick={() => onSelect(article)}>
      <span className="article-kicker">
        {article.entity_class}
        {article.year ? " · " + Math.abs(article.year) : ""}
      </span>
      <strong>{article.title}</strong>
      <span className="article-summary">{article.first_paragraph}</span>
      <span className="match-reason">{article.match_reason} →</span>
    </button>
  );
}

export default function App() {
  const [query, setQuery] = useState("Ann Arbor, Michigan");
  const [radius, setRadius] = useState(50);
  const [data, setData] = useState<SearchResponse | null>(null);
  const [mainTab, setMainTab] = useState<MainTab>("map");
  const [railTab, setRailTab] = useState<RailTab>("connected");
  const [selected, setSelected] = useState<ArticleCard | null>(null);
  const [detail, setDetail] = useState<ArticleDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function runSearch(searchQuery = query) {
    setLoading(true);
    setError("");
    setSelected(null);
    try {
      setData(await searchHistory(searchQuery, radius));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void runSearch("Ann Arbor, Michigan");
  }, []);

  useEffect(() => {
    if (!selected) {
      setDetail(null);
      return;
    }
    void fetchArticle(selected.page_id).then(setDetail).catch(() => setDetail(null));
  }, [selected]);

  const allArticles = useMemo(
    () => [...(data?.mapped ?? []), ...(data?.connected ?? [])],
    [data],
  );
  const timeline = useMemo(
    () =>
      allArticles
        .filter((article) => article.year !== null)
        .sort((a, b) => (a.year ?? 0) - (b.year ?? 0)),
    [allArticles],
  );
  const railArticles = railTab === "mapped" ? data?.mapped : data?.connected;

  function submit(event: FormEvent) {
    event.preventDefault();
    void runSearch();
  }

  return (
    <div className="shell">
      <header className="masthead">
        <div>
          <p className="eyebrow">Explore place through time</p>
          <h1>Local History <em>Atlas</em></h1>
        </div>
        <form className="search" onSubmit={submit}>
          <label>
            <span>City</span>
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search a city"
            />
          </label>
          <label className="radius">
            <span>Radius</span>
            <select
              value={radius}
              onChange={(event) => setRadius(Number(event.target.value))}
            >
              <option value={10}>10 km</option>
              <option value={25}>25 km</option>
              <option value={50}>50 km</option>
              <option value={100}>100 km</option>
            </select>
          </label>
          <button type="submit" disabled={loading}>
            {loading ? "Searching…" : "Explore"}
          </button>
        </form>
      </header>

      <nav className="main-tabs" aria-label="Views">
        {(["map", "stories", "timeline"] as MainTab[]).map((tab) => (
          <button
            className={mainTab === tab ? "active" : ""}
            onClick={() => setMainTab(tab)}
            key={tab}
          >
            {tab}
          </button>
        ))}
      </nav>

      {error && <div className="error">{error}</div>}

      {data && (
        <>
          <section className="place-heading">
            <div>
              <p className="eyebrow">Current place</p>
              <h2>{data.query.label.split(",").slice(0, 2).join(",")}</h2>
            </div>
            <div className="counts">
              <span><b>{data.counts.mapped}</b> mapped</span>
              <span><b>{data.counts.connected}</b> connected</span>
            </div>
          </section>

          {mainTab === "map" && (
            <main className="atlas-grid">
              <MapView
                center={[data.query.longitude, data.query.latitude]}
                articles={data.mapped}
                onSelect={setSelected}
              />
              <aside className="story-rail">
                <div className="rail-tabs">
                  <button
                    className={railTab === "mapped" ? "active" : ""}
                    onClick={() => setRailTab("mapped")}
                  >
                    Mapped <span>{data.counts.mapped}</span>
                  </button>
                  <button
                    className={railTab === "connected" ? "active" : ""}
                    onClick={() => setRailTab("connected")}
                  >
                    Connected <span>{data.counts.connected}</span>
                  </button>
                </div>
                <p className="rail-note">
                  {railTab === "mapped"
                    ? "Stories with a known geographic location."
                    : "Stories linked by text, kept off the map until their location is known."}
                </p>
                <div className="article-list">
                  {(railArticles ?? []).slice(0, 15).map((article) => (
                    <ArticleRow
                      article={article}
                      onSelect={setSelected}
                      key={article.page_id}
                    />
                  ))}
                </div>
              </aside>
            </main>
          )}

          {mainTab === "stories" && (
            <main className="story-grid">
              {allArticles.map((article) => (
                <ArticleRow
                  article={article}
                  onSelect={setSelected}
                  key={article.page_id}
                />
              ))}
            </main>
          )}

          {mainTab === "timeline" && (
            <main className="timeline">
              {timeline.map((article) => (
                <button onClick={() => setSelected(article)} key={article.page_id}>
                  <time>{article.year && Math.abs(article.year)}</time>
                  <span>
                    <b>{article.title}</b>
                    <small>{article.entity_class}</small>
                  </span>
                </button>
              ))}
            </main>
          )}
        </>
      )}

      {selected && (
        <div className="drawer-backdrop" onClick={() => setSelected(null)}>
          <article className="drawer" onClick={(event) => event.stopPropagation()}>
            <button className="drawer-close" onClick={() => setSelected(null)}>
              Close
            </button>
            <p className="eyebrow">{selected.match_reason}</p>
            <h2>{selected.title}</h2>
            <p className="drawer-meta">
              {selected.entity_class}
              {selected.year ? " · " + Math.abs(selected.year) : ""}
              {selected.country ? " · " + selected.country : ""}
            </p>
            <p className="lead">{detail?.first_paragraph ?? selected.first_paragraph}</p>
            {detail ? (
              <div className="article-text">
                {detail.full_text
                  .split(/\n{2,}/)
                  .slice(1)
                  .map((paragraph, index) => <p key={index}>{paragraph}</p>)}
              </div>
            ) : (
              <p className="loading-copy">Loading article…</p>
            )}
          </article>
        </div>
      )}
    </div>
  );
}
