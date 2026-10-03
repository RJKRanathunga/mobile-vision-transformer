from PIL import Image

from mvt.data.input import MVTInputProcessor

def test_mvt_input_shapes(tmp_path):
    processor = MVTInputProcessor()

    template_image = Image.new("RGB", (300, 200))
    search_image = Image.new("RGB", (500, 400))

    template_path = tmp_path / "template.jpg"
    search_path = tmp_path / "search.jpg"

    template_image.save(template_path)
    search_image.save(search_path)

    template, search = processor.load_pair(
        template_path, search_path
    )

    assert template.shape == (1,3,128,128)
    assert search.shape == (1,3,256,256)