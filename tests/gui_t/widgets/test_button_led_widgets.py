"""Tests for reusable image-capable button and LED widgets."""

from typing import Any

from sonicrack.gui.widgets.button_widget import ImageButtonStyle, ImagePushButton
from sonicrack.gui.widgets.led_widget import LedIndicator, LedStyle


def test_image_push_button_uses_style_size(qapp: Any):
    del qapp
    button = ImagePushButton(style=ImageButtonStyle(size=18), text="A")

    assert button.isCheckable()
    assert button.width() == 18
    assert button.height() == 18
    assert button.text() == "A"


def test_image_push_button_toggles(qapp: Any):
    del qapp
    button = ImagePushButton(style=ImageButtonStyle(size=20))

    button.setChecked(True)

    assert button.isChecked()


def test_image_push_button_loads_packaged_resource(qapp: Any):
    del qapp
    button = ImagePushButton(
        style=ImageButtonStyle(off_resource=("icons", "icon.png"), size=20)
    )

    assert not button.icon().isNull()


def test_led_indicator_tracks_state(qapp: Any):
    del qapp
    led = LedIndicator(style=LedStyle(size=10))

    assert not led.is_on()
    assert led.width() == 10
    assert led.height() == 10

    led.set_on(True)

    assert led.is_on()


def test_led_loads_packaged_skin(qapp: Any):
    del qapp
    from sonicrack.gui.widgets.skin import skin_available

    led = LedIndicator(style=LedStyle(size=12))
    if skin_available():
        assert led._on_pixmap is not None
        assert led._off_pixmap is not None
