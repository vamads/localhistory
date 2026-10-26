import pandas as pd

from localhistory.search import article_quality_features


def test_metadata_completeness_is_normalized_by_entity_type():
    frame = pd.DataFrame(
        [
            {
                "entity_class": "person",
                "wikidata_id": "Q1",
                "instance_of": '["human"]',
                "occupation": "lawyer",
                "birth_date": "1758-05-06",
                "death_date": "1794-07-28",
                "country": None,
                "location": None,
                "lat": None,
                "lon": None,
                "inception": None,
                "point_in_time": None,
                "start_time": None,
                "end_time": None,
                "is_list_article": False,
                "first_paragraph": "A substantial historical introduction.",
                "full_text": "A substantial historical article.",
            },
            {
                "entity_class": "place",
                "wikidata_id": "Q2",
                "instance_of": '["city"]',
                "occupation": None,
                "birth_date": None,
                "death_date": None,
                "country": "France",
                "location": "France",
                "lat": 48.8,
                "lon": 2.3,
                "inception": "-0052",
                "point_in_time": None,
                "start_time": None,
                "end_time": None,
                "is_list_article": False,
                "first_paragraph": "A substantial historical introduction.",
                "full_text": "A substantial historical article.",
            },
        ]
    )

    result = article_quality_features(frame)

    assert result.loc[0, "metadata_completeness"] > 0.8
    assert result.loc[1, "metadata_completeness"] > 0.8
