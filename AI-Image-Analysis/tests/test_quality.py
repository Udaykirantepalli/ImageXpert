from src.image_quality import assess_quality
from src.image_analysis import analyze_image
from src.recommendations import generate_recommendations
def test_quality_and_recommendations(rgb_image):
    quality = assess_quality(rgb_image); assert 0 <= quality["quality_score"] <= 100
    items = generate_recommendations(analyze_image(rgb_image), quality, 4.5); assert len(items) == 5 and "suggested_pipeline" in items[0]
