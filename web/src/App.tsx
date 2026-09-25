import { FormEvent, useEffect, useMemo, useState } from "react";
import { fetchArticle, fetchCoordinates, searchHistory } from "./api";
import MapView from "./MapView";
import type { ArticleCard, ArticleDetail, MapPoint, SearchResponse } from "./types";

type MainTab = "connected" | "map" | "timeline";

function ArticleRow({
  article,
  onSelect,
  parchment = false,
  rank,
}: {
  article: ArticleCard;
  onSelect: (article: ArticleCard) => void;
  parchment?: boolean;
  rank?: number;
}) {
  if (parchment) {
    return (
      <button type="button" className="article-row parchment-card" onClick={() => onSelect(article)}>
        <span className="archive-header">
          <span className="archive-index">{String(rank ?? 0).padStart(2, "0")}</span>
          <span className="article-kicker">{article.entity_class}</span>
          <span className="archive-score">{article.score.toFixed(1)}</span>
        </span>
        <span className="archive-rule" />
        <strong>{article.title}</strong>
        <span className="article-summary">{article.first_paragraph}</span>
        <span className="archive-footer">
          <span>{article.match_reason}</span>
          {article.year && <span>{Math.abs(article.year)}</span>}
        </span>
      </button>
    );
  }

  return (
    <button
      type="button"
      className="article-row"
      onClick={() => onSelect(article)}
    >
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

function EmptyState({ title, copy }: { title: string; copy: string }) {
  return (
    <section className="empty-state" aria-live="polite">
      <span className="empty-mark" aria-hidden="true">—</span>
      <h3>{title}</h3>
      <p>{copy}</p>
    </section>
  );
}

export default function App() {
  const [query, setQuery] = useState("Ann Arbor, Michigan");
  const [radius, setRadius] = useState(50);
  const [data, setData] = useState<SearchResponse | null>(null);
  const [mainTab, setMainTab] = useState<MainTab>("connected");
  const [selected, setSelected] = useState<ArticleCard | null>(null);
  const [detail, setDetail] = useState<ArticleDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [mapPoints, setMapPoints] = useState<MapPoint[]>([]);

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
    if (mainTab !== "map" || mapPoints.length > 0) return;
    void fetchCoordinates().then(setMapPoints).catch(() => {
      setError("Could not load map coordinates");
    });
  }, [mainTab, mapPoints.length]);

  useEffect(() => {
    if (!selected) {
      setDetail(null);
      return;
    }
    void fetchArticle(selected.page_id)
      .then((article) => {
        setDetail(article);
        setSelected((current) => current ? {
          ...current,
          title: article.title,
          first_paragraph: article.first_paragraph,
          entity_class: article.entity_class,
          year: article.year,
          country: article.country,
        } : current);
      })
      .catch(() => setDetail(null));
  }, [selected]);

  const allArticles = useMemo(
    () => data?.connected ?? [],
    [data],
  );
  const timeline = useMemo(
    () =>
      allArticles
        .filter((article) => article.year !== null)
        .sort((a, b) => (a.year ?? 0) - (b.year ?? 0)),
    [allArticles],
  );
  const connectedArticles = useMemo(
    () => data?.connected ?? [],
    [data],
  );
  const mappedArticles = useMemo(
    () => data?.mapped ?? [],
    [data],
  );

  function selectMapPage(pageId: number) {
    const result = allArticles.find((article) => article.page_id === pageId);
    if (result) {
      setSelected(result);
      return;
    }
    setSelected({
      page_id: pageId,
      title: "Loading article…",
      first_paragraph: "",
      entity_class: "other",
      year: null,
      country: null,
      latitude: null,
      longitude: null,
      distance_km: null,
      score: 0,
      source: "coordinates",
      match_reason: "Mapped nearby",
    });
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    void runSearch();
  }

  return (
    <div className="shell">
      <header className="masthead">
        <div className="library-brand">
          <span className="library-seal" aria-hidden="true">LH</span>
          <div>
            <p className="eyebrow">A field guide to local memory</p>
            <h1>Local History <em>Atlas</em></h1>
            <p className="masthead-deck">
              Find people, places, and events connected to a city.
            </p>
          </div>
        </div>
        <form className="search" onSubmit={submit}>
          <label>
            <span>Location</span>
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search a city or place"
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
            {loading ? "Searching…" : "Search"}
          </button>
        </form>
      </header>

      <nav className="main-tabs" aria-label="Views">
        {([
          ["connected", "Related"],
          ["map", "Map"],
          ["timeline", "Timeline"],
        ] as [MainTab, string][]).map(([tab, label]) => (
          <button
            type="button"
            className={mainTab === tab ? "active" : ""}
            onClick={() => setMainTab(tab)}
            aria-current={mainTab === tab ? "page" : undefined}
            key={tab}
          >
            {label}
          </button>
        ))}
      </nav>

      {error && <div className="error" role="alert">{error}</div>}

      {data && (
        <>
          <section className="place-heading">
            <div>
              <p className="eyebrow">Local history near</p>
              <h2>{data.query.label.split(",").slice(0, 2).join(",")}</h2>
            </div>
            <span className="heading-ornament" aria-hidden="true">❦</span>
            <div className="counts">
              <span><b>{data.counts.mapped}</b> mapped locations</span>
              <span><b>{data.counts.connected}</b> related stories</span>
            </div>
          </section>

          {loading && <div className="results-status" role="status">Updating results…</div>}

          {mainTab === "map" && (
            <main className="atlas-grid">
              <MapView
                center={[data.query.longitude, data.query.latitude]}
                points={mapPoints}
                radiusKm={radius}
                onSelect={selectMapPage}
              />
              <aside className="story-rail">
                <div className="rail-heading">
                  <span>Mapped stories</span>
                  <b>{data.counts.mapped}</b>
                </div>
                <p className="rail-note">
                  Stories with known locations.
                </p>
                <div className="article-list">
                  {mappedArticles.length > 0 ? mappedArticles.slice(0, 15).map((article) => (
                    <ArticleRow article={article} onSelect={setSelected} key={article.page_id} />
                  )) : (
                    <EmptyState title="No mapped stories" copy="Try a larger radius." />
                  )}
                </div>
              </aside>
            </main>
          )}

          {mainTab === "connected" && (
            <main className="story-grid">
              {connectedArticles.length > 0 ? connectedArticles.map((article, index) => (
                <ArticleRow
                  article={article}
                  onSelect={setSelected}
                  parchment
                  rank={index + 1}
                  key={article.page_id}
                />
              )) : (
                <EmptyState title="No related stories" copy="Try another location or a broader search." />
              )}
            </main>
          )}

          {mainTab === "timeline" && (
            <main className="timeline">
              {timeline.length > 0 ? timeline.map((article) => (
                <button type="button" onClick={() => setSelected(article)} key={article.page_id}>
                  <time>{article.year && Math.abs(article.year)}</time>
                  <span>
                    <b>{article.title}</b>
                    <small>{article.entity_class}</small>
                  </span>
                </button>
              )) : (
                <EmptyState title="No dated stories" copy="This search has no timeline entries." />
              )}
            </main>
          )}
        </>
      )}

      {selected && (
        <div className="drawer-backdrop" onClick={() => setSelected(null)}>
          <article className="drawer" role="dialog" aria-modal="true" aria-labelledby="article-title" onClick={(event) => event.stopPropagation()}>
            <button type="button" className="drawer-close" onClick={() => setSelected(null)}>
              Close
            </button>
            <p className="eyebrow">{selected.match_reason}</p>
            <h2 id="article-title">{selected.title}</h2>
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
