import pytest

from soccer_highlights import review_overlay
from soccer_highlights.config import CropConfig, ReviewOverlayConfig
from soccer_highlights.review_overlay import (
    OverlayLedger,
    build_overlay_graph,
    clean_signature,
    dot_frame_window,
    overlay_signature,
    publish_file,
    step_is_done,
    write_dot_png,
)

CROP = CropConfig()
OVL = ReviewOverlayConfig()


def test_graph_is_none_with_nothing_to_draw():
    assert build_overlay_graph(1280, 720, None, None, 30.0, CROP, OVL) is None


def test_box_only_graph_matches_the_export_crop_geometry():
    graph = build_overlay_graph(1280, 720, "left", None, 30.0, CROP, OVL)
    # 2/3 x 2/3 of 1280x720, left edge, y 1/6..5/6
    assert graph.startswith("drawbox=x=0:y=120:w=853:h=480:")
    assert "t=3" in graph and "[vout]" not in graph
    right = build_overlay_graph(1280, 720, "right", None, 30.0, CROP, OVL)
    assert right.startswith("drawbox=x=427:y=120:w=853:h=480:")
    center = build_overlay_graph(1280, 720, "center", None, 30.0, CROP, OVL)
    assert center.startswith("drawbox=x=213:y=120:w=853:h=480:")


def test_box_thickness_scales_with_width():
    assert "t=6" in build_overlay_graph(2560, 1440, "left", None, 30.0, CROP, OVL)


def test_dot_window_is_20_frames_centred_and_truncated_at_the_clip_start():
    start, end = dot_frame_window(5.0, 20, 30.0)
    assert end - start == pytest.approx(20 / 30) and (start + end) / 2 == pytest.approx(5.0)
    assert dot_frame_window(0.1, 20, 30.0)[0] == 0.0


def test_dot_only_graph_overlays_top_right_for_the_window():
    graph = build_overlay_graph(1280, 720, None, 5.0, 30.0, CROP, OVL)
    assert graph.startswith("[0:v][1:v]overlay=x=main_w-overlay_w-15:y=15:")  # 1.2% margin of 1280
    assert "gte(t,4.400)*lt(t,5.600)" in graph and graph.endswith("[vout]")
    assert "drawbox" not in graph  # blank action_region -> no box


def test_box_and_dot_graph_chains_both():
    graph = build_overlay_graph(1280, 720, "center", 5.0, 30.0, CROP, OVL)
    assert graph.startswith("[0:v]drawbox=") and "[bx];[bx][1:v]overlay=" in graph


def test_dot_png_has_the_requested_size_and_a_solid_centre(tmp_path):
    from PIL import Image

    path = tmp_path / "dot.png"
    write_dot_png(path, 19, "FFD600")
    im = Image.open(path).convert("RGBA")
    assert im.size == (19, 19)
    assert im.getpixel((9, 9)) == (255, 214, 0, 255)  # solid fill
    assert im.getpixel((0, 0))[3] == 0  # corner transparent


def test_overlay_signature_changes_with_anything_that_changes_the_pixels():
    base = overlay_signature(10.0, 25.0, "left", 5.0, OVL, CROP)
    assert base == overlay_signature(10.0, 25.0, "left", 5.0, OVL, CROP)
    assert base != overlay_signature(10.0, 25.0, None, 5.0, OVL, CROP)  # region arrived late
    assert base != overlay_signature(10.0, 25.0, "left", 5.1, OVL, CROP)
    assert base != overlay_signature(10.0, 25.0, "left", 5.0, ReviewOverlayConfig(dot_frames=10), CROP)
    assert base != overlay_signature(10.0, 25.0, "left", 5.0, OVL, CropConfig(y_center=0.6))
    assert clean_signature(10.0, 25.0) != clean_signature(10.0, 26.0)


def test_ledger_persists_and_a_step_needs_both_signature_and_a_playable_file(tmp_path, monkeypatch):
    ledger_path = tmp_path / "ledger.json"
    ledger = OverlayLedger(ledger_path)
    ledger.record("overlaid", "clip_001.mp4", "sig-A")
    reloaded = OverlayLedger(ledger_path)
    clip = tmp_path / "clip_001.mp4"
    clip.write_bytes(b"x")

    monkeypatch.setattr(review_overlay, "is_playable", lambda path: True)
    assert step_is_done(reloaded, "overlaid", "clip_001.mp4", "sig-A", clip)
    assert not step_is_done(reloaded, "overlaid", "clip_001.mp4", "sig-B", clip)  # inputs changed
    assert not step_is_done(reloaded, "clean", "clip_001.mp4", "sig-A", clip)  # different step

    monkeypatch.setattr(review_overlay, "is_playable", lambda path: False)  # killed-encode corpse
    assert not step_is_done(reloaded, "overlaid", "clip_001.mp4", "sig-A", clip)

    reloaded.forget("overlaid", "clip_001.mp4")
    assert not reloaded.matches("overlaid", "clip_001.mp4", "sig-A")


def test_unreadable_ledger_means_nothing_is_done(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text("{not json", encoding="utf-8")
    assert not OverlayLedger(path).matches("clean", "clip_001.mp4", "x")


def test_publish_file_replaces_atomically_and_leaves_no_part_file(tmp_path):
    src = tmp_path / "work" / "clip_001.mp4"
    src.parent.mkdir()
    src.write_bytes(b"new")
    dst = tmp_path / "Review" / "clip_001.mp4"
    dst.parent.mkdir()
    dst.write_bytes(b"old")
    publish_file(src, dst)
    assert dst.read_bytes() == b"new"
    assert sorted(p.name for p in dst.parent.iterdir()) == ["clip_001.mp4"]  # Review stays flat and clean
