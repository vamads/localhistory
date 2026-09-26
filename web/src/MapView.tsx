import { useEffect, useMemo, useRef } from "react";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import type { MapPoint } from "./types";
import type {
  GeoJSONSource,
  Map as MapLibreMap,
  StyleSpecification,
} from "maplibre-gl";

type Props = {
  center: [number, number];
  points: MapPoint[];
  radiusKm: number;
  onSelect: (pageId: number) => void;
};

const mapStyle: StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      attribution: "© OpenStreetMap contributors",
    },
  },
  layers: [{ id: "osm", type: "raster", source: "osm" }],
};

function distanceKm(center: [number, number], point: MapPoint) {
  const [lon1, lat1] = center;
  const radians = Math.PI / 180;
  const dLat = (point.latitude - lat1) * radians;
  const dLon = (point.longitude - lon1) * radians;
  const a = Math.sin(dLat / 2) ** 2
    + Math.cos(lat1 * radians)
      * Math.cos(point.latitude * radians)
      * Math.sin(dLon / 2) ** 2;
  return 6371 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function articleGeoJson(points: MapPoint[]) {
  return {
    type: "FeatureCollection" as const,
    features: points.map((point) => ({
        type: "Feature" as const,
        geometry: {
          type: "Point" as const,
          coordinates: [point.longitude, point.latitude],
        },
        properties: {
          page_id: point.page_id,
        },
      })),
  };
}

export default function MapView({ center, points, radiusKm, onSelect }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<MapLibreMap | null>(null);
  const onSelectRef = useRef(onSelect);

  const [centerLon, centerLat] = center;
  const visiblePoints = useMemo(
    () => points.filter((point) => distanceKm([centerLon, centerLat], point) <= radiusKm),
    [centerLon, centerLat, points, radiusKm],
  );

  onSelectRef.current = onSelect;

  useEffect(() => {
    let disposed = false;

    void import("maplibre-gl").then((maplibre) => {
      if (disposed || !container.current) return;

      maplibre.setWorkerUrl(workerUrl);
      const instance = new maplibre.Map({
        container: container.current,
        style: mapStyle,
        center,
        zoom: 10,
      });
      map.current = instance;
      instance.addControl(new maplibre.NavigationControl(), "top-left");

      new maplibre.Marker({ color: "#b6643e" })
        .setLngLat(center)
        .setPopup(new maplibre.Popup().setText("Search location"))
        .addTo(instance);

      instance.on("load", () => {
        instance.addSource("articles", {
          type: "geojson",
          data: articleGeoJson(visiblePoints),
          cluster: true,
          clusterMaxZoom: 14,
          clusterRadius: 44,
        });
        instance.addLayer({
          id: "clusters",
          type: "circle",
          source: "articles",
          filter: ["has", "point_count"],
          paint: {
            "circle-color": "#243a35",
            "circle-radius": ["step", ["get", "point_count"], 18, 10, 24, 40, 30],
            "circle-stroke-color": "#f7f4ed",
            "circle-stroke-width": 2,
          },
        });
        instance.addLayer({
          id: "cluster-count",
          type: "symbol",
          source: "articles",
          filter: ["has", "point_count"],
          layout: {
            "text-field": ["get", "point_count_abbreviated"],
            "text-size": 12,
          },
          paint: { "text-color": "#ffffff" },
        });
        instance.addLayer({
          id: "article-points",
          type: "circle",
          source: "articles",
          filter: ["!", ["has", "point_count"]],
          paint: {
            "circle-color": "#d49b57",
            "circle-radius": 7,
            "circle-stroke-color": "#243a35",
            "circle-stroke-width": 1.5,
          },
        });

        instance.on("click", "clusters", async (event) => {
          const feature = instance.queryRenderedFeatures(event.point, {
            layers: ["clusters"],
          })[0];
          const clusterId = feature?.properties?.cluster_id;
          const source = instance.getSource("articles") as GeoJSONSource;
          if (clusterId === undefined) return;
          const zoom = await source.getClusterExpansionZoom(clusterId);
          if (feature.geometry.type !== "Point") return;
          const coordinates = feature.geometry.coordinates as [number, number];
          instance.easeTo({ center: coordinates, zoom });
        });

        instance.on("click", "article-points", (event) => {
          const pageId = Number(event.features?.[0]?.properties?.page_id);
          if (Number.isFinite(pageId)) onSelectRef.current(pageId);
        });

        for (const layer of ["clusters", "article-points"]) {
          instance.on("mouseenter", layer, () => {
            instance.getCanvas().style.cursor = "pointer";
          });
          instance.on("mouseleave", layer, () => {
            instance.getCanvas().style.cursor = "";
          });
        }
      });
    });

    return () => {
      disposed = true;
      map.current?.remove();
      map.current = null;
    };
  }, [visiblePoints, center[0], center[1]]);

  return <div className="map" ref={container} aria-label="Map of article locations" />;
}
