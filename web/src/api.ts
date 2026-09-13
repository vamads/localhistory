import type { ArticleDetail, SearchResponse } from "./types";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function request<T>(path: string): Promise<T> {
  const response = await fetch(API_URL + path);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? "Request failed (" + response.status + ")");
  }
  return response.json() as Promise<T>;
}

export function searchHistory(query: string, radiusKm: number) {
  const params = new URLSearchParams({
    q: query,
    radius_km: String(radiusKm),
  });
  return request<SearchResponse>("/api/search?" + params);
}

export function fetchArticle(pageId: number) {
  return request<ArticleDetail>("/api/articles/" + pageId);
}
