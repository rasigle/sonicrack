"""Generate AudioPlayground application icon."""

import os

import numpy as np
from PIL import Image, ImageDraw


def create_app_icon(size=512):
    """Create a modern, professional icon for AudioPlayground.

    Design concept: Modular synth patch cables with waveform
    - Dark background
    - Colorful gradient cables
    - Sine wave integrated into design
    - Modern, sleek appearance
    """
    # Create image with dark background
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background - dark rounded square with gradient effect
    margin = size // 10
    bg_rect = [margin, margin, size - margin, size - margin]

    # Draw background with gradient simulation (multiple overlapping rounded rectangles)
    for i in range(20):
        alpha = int(255 * (1 - i / 20))
        shade = int(30 + i * 3)
        color = (shade, shade, shade + 10, alpha)
        offset = i * 2
        draw.rounded_rectangle(
            [
                bg_rect[0] + offset,
                bg_rect[1] + offset,
                bg_rect[2] - offset,
                bg_rect[3] - offset,
            ],
            radius=size // 8,
            fill=color,
        )

    # Main background
    draw.rounded_rectangle(bg_rect, radius=size // 8, fill=(25, 25, 30, 255))

    # Draw modular patch cables (colorful bezier-like curves)
    cable_colors = [
        (255, 100, 100),  # Red
        (100, 200, 255),  # Blue
        (100, 255, 150),  # Green
        (255, 200, 100),  # Yellow/Orange
        (200, 100, 255),  # Purple
    ]

    center_x = size // 2
    center_y = size // 2
    radius_base = size // 3

    # Draw cables as arcs connecting different points
    for i, color in enumerate(cable_colors):
        angle_start = (i * 72) * np.pi / 180  # 360/5 = 72 degrees apart
        angle_end = ((i + 2) * 72) * np.pi / 180

        # Start and end points on a circle
        x1 = center_x + int(radius_base * np.cos(angle_start))
        y1 = center_y + int(radius_base * np.sin(angle_start))
        x2 = center_x + int(radius_base * np.cos(angle_end))
        y2 = center_y + int(radius_base * np.sin(angle_end))

        # Draw cable with glow effect
        for width_offset in range(3, 0, -1):
            alpha = int(100 + (3 - width_offset) * 50)
            glow_color = (*color, alpha)
            draw.line(
                [x1, y1, x2, y2], fill=glow_color, width=size // 40 + width_offset * 2
            )

        # Draw connector dots at ends
        dot_size = size // 30
        for x, y in [(x1, y1), (x2, y2)]:
            # Outer glow
            draw.ellipse(
                [
                    x - dot_size * 1.5,
                    y - dot_size * 1.5,
                    x + dot_size * 1.5,
                    y + dot_size * 1.5,
                ],
                fill=(*color, 100),
            )
            # Inner dot
            draw.ellipse(
                [x - dot_size, y - dot_size, x + dot_size, y + dot_size],
                fill=(*color, 255),
            )

    # Draw central waveform (sine wave in a circle)
    wave_points = []
    num_points = 200
    wave_radius = size // 6

    for i in range(num_points + 1):
        angle = (i / num_points) * 2 * np.pi
        # Sine wave modulation
        wave_amp = wave_radius * 0.3 * np.sin(angle * 3)
        r = wave_radius + wave_amp
        x = center_x + r * np.cos(angle)
        y = center_y + r * np.sin(angle)
        wave_points.append((x, y))

    # Draw waveform with gradient glow
    for width in [12, 8, 4]:
        alpha = int(255 * (5 - width) / 5)
        draw.line(wave_points, fill=(100, 200, 255, alpha), width=width, joint="curve")

    # Draw center circle with gradient
    inner_radius = size // 12
    for i in range(15):
        r = inner_radius - i
        alpha = int(255 * (15 - i) / 15)
        draw.ellipse(
            [center_x - r, center_y - r, center_x + r, center_y + r],
            fill=(50, 150, 255, alpha),
        )

    # Add subtle outer glow
    glow_radius = size // 2 - margin
    for i in range(20):
        r = glow_radius + i * 3
        alpha = int(30 * (20 - i) / 20)
        draw.ellipse(
            [center_x - r, center_y - r, center_x + r, center_y + r],
            outline=(100, 180, 255, alpha),
        )

    return img


def save_icon_sizes(base_image, output_dir="."):
    """Save icon in multiple sizes for different platforms."""
    os.makedirs(output_dir, exist_ok=True)

    sizes = {
        "icon_16.png": 16,
        "icon_32.png": 32,
        "icon_48.png": 48,
        "icon_64.png": 64,
        "icon_128.png": 128,
        "icon_256.png": 256,
        "icon_512.png": 512,
        "icon.png": 512,  # Main icon
    }

    for filename, size in sizes.items():
        resized = base_image.resize((size, size), Image.Resampling.LANCZOS)
        filepath = os.path.join(output_dir, filename)
        resized.save(filepath, "PNG")
        print(f"✅ Created: {filepath}")

    # Create Windows ICO file (multi-resolution)
    ico_path = os.path.join(output_dir, "icon.ico")
    base_image.save(
        ico_path,
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print(f"✅ Created: {ico_path}")


if __name__ == "__main__":
    print("🎨 Generating AudioPlayground icon...")

    # Create high-resolution base icon
    icon = create_app_icon(size=512)

    # Save in multiple sizes
    save_icon_sizes(icon)

    print("\n🎉 Icon generation complete!")
    print("\nGenerated files:")
    print("  - icon.png (512x512 - main icon)")
    print("  - icon.ico (Windows icon with multiple sizes)")
    print("  - icon_16.png to icon_512.png (various sizes)")
    print("\nTo use in your app:")
    print("  1. Copy 'resources' folder to your project root")
    print(
        "  2. In main_window.py, add: self.setWindowIcon(QIcon('resources/icon.png'))"
    )
    print("  3. For packaging, reference icon.ico in your build config")
