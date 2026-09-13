export type ArticleCard = {
  page_id: number;
  title: string;
  first_paragraph: string;
  entity_class: string;
  year: number | null;
  country: string | null;
  latitude: number | null;
  longitude: number | null;
  distance_km: number | null;
  score: number;
  source: string;
  match_reason: string;
};

export type SearchResponse = {
  query: {
    label: string;
    latitude: number;
    longitude: number;
  };
  mapped: ArticleCard[];
  connected: ArticleCard[];
  counts: { mapped: number; connected: number };
};

export type ArticleDetail = {
  page_id: number;
  title: string;
  first_paragraph: string;
  full_text: string;
  entity_class: string;
  year: number | null;
  country: string | null;
};
