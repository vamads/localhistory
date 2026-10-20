from localhistory.text import article_lead, clean_article_text


def test_image_markup_does_not_appear_in_article_text():
    image_remnant = (
        "250px|thumb|December 13, 1937: Japanese Army captures Nanking"
    )
    full_text = image_remnant + "\n\nThe Battle of Nanjing began here."

    assert clean_article_text(full_text) == "The Battle of Nanjing began here."
    assert article_lead(image_remnant, full_text) == "The Battle of Nanjing began here."
