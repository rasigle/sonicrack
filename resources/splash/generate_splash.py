"""Generate AudioPlayground splash screen."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_splash_screen(width=800, height=600):
    """Create a stunning splash screen for AudioPlayground.

    Design concept: Dark modular synth aesthetic with:
    - Animated-looking patch cables
    - Glowing waveforms
    - Large logo/title
    - Version info
    - Professional gradient background
    """
    # Create image with dark gradient background
    img = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)

    # Create dark gradient background
    for y in range(height):
        # Gradient from dark blue-gray at top to darker at bottom
        progress = y / height
        r = int(15 + progress * 10)
        g = int(20 + progress * 10)
        b = int(35 + progress * 15)
        draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

    # Draw decorative patch cables in background
    cable_colors = [
        (255, 100, 100, 150),   # Red
        (100, 200, 255, 150),   # Blue
        (100, 255, 150, 150),   # Green
        (255, 200, 100, 150),   # Yellow
        (200, 100, 255, 150),   # Purple
    ]

    # Left side cables (flowing from top-left)
    for i, color in enumerate(cable_colors):
        start_x = 50 + i * 30
        start_y = 50

        # Create flowing bezier-like curves
        points = []
        num_points = 100
        for t in range(num_points + 1):
            t_norm = t / num_points
            # Bezier curve simulation
            x = start_x + t_norm * (width * 0.3) + np.sin(t_norm * 3 * np.pi) * 50
            y = start_y + t_norm * (height - 100) + np.cos(t_norm * 2 * np.pi) * 30
            points.append((x, y))

        # Draw cable with glow
        for width_offset in range(4, 0, -1):
            alpha = int(color[3] * (5 - width_offset) / 5)
            glow_color = (color[0], color[1], color[2], alpha)
            draw.line(points, fill=glow_color, width=3 + width_offset, joint='curve')

    # Right side cables (flowing from top-right)
    for i, color in enumerate(cable_colors):
        start_x = width - 50 - i * 30
        start_y = 50

        points = []
        num_points = 100
        for t in range(num_points + 1):
            t_norm = t / num_points
            x = start_x - t_norm * (width * 0.3) - np.sin(t_norm * 3 * np.pi) * 50
            y = start_y + t_norm * (height - 100) - np.cos(t_norm * 2 * np.pi) * 30
            points.append((x, y))

        for width_offset in range(4, 0, -1):
            alpha = int(color[3] * (5 - width_offset) / 5)
            glow_color = (color[0], color[1], color[2], alpha)
            draw.line(points, fill=glow_color, width=3 + width_offset, joint='curve')

    # Draw central waveforms (multiple overlapping)
    center_x = width // 2
    waveform_y = height // 2

    # Multiple waveforms with different frequencies
    waveform_colors = [
        (100, 200, 255, 200),   # Blue
        (255, 100, 150, 180),   # Pink
        (100, 255, 200, 160),   # Cyan
    ]

    for wave_idx, color in enumerate(waveform_colors):
        points = []
        num_points = 300
        wave_width = width * 0.7
        wave_height = 60

        for i in range(num_points + 1):
            t = i / num_points
            x = center_x - wave_width / 2 + t * wave_width
            # Multiple sine waves with different frequencies
            y = waveform_y + np.sin(t * (3 + wave_idx) * 2 * np.pi) * wave_height / (wave_idx + 1)
            y += np.sin(t * (5 + wave_idx * 2) * 2 * np.pi) * wave_height / (wave_idx + 2) / 2
            points.append((x, y))

        # Draw with glow effect
        for width_val in [8, 5, 3]:
            alpha = int(color[3] * (9 - width_val) / 9)
            glow_color = (color[0], color[1], color[2], alpha)
            draw.line(points, fill=glow_color, width=width_val, joint='curve')

    # Draw connector nodes/dots along the bottom
    num_dots = 15
    for i in range(num_dots):
        x = width * 0.1 + (width * 0.8) * i / (num_dots - 1)
        y = height - 80

        # Choose color based on position
        color_idx = i % len(cable_colors)
        color = cable_colors[color_idx]

        # Outer glow
        for r in [12, 8, 5]:
            alpha = int(150 * (13 - r) / 13)
            draw.ellipse(
                [x - r, y - r, x + r, y + r],
                fill=(color[0], color[1], color[2], alpha)
            )

    # Add central glowing circle (logo background)
    logo_y = height * 0.25
    for r in range(100, 0, -2):
        alpha = int(80 * (100 - r) / 100)
        draw.ellipse(
            [center_x - r, logo_y - r, center_x + r, logo_y + r],
            fill=(50, 120, 200, alpha)
        )

    # Try to load a nice font, fallback to default
    try:
        # Try to find a modern font
        title_font = ImageFont.truetype("arial.ttf", 72)
        subtitle_font = ImageFont.truetype("arial.ttf", 32)
        version_font = ImageFont.truetype("arial.ttf", 24)
        loading_font = ImageFont.truetype("arial.ttf", 20)
    except:
        # Fallback to default font
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        version_font = ImageFont.load_default()
        loading_font = ImageFont.load_default()

    # Draw main title with glow effect
    title = "AudioPlayground"

    # Calculate text size and position for centering
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (width - title_width) // 2
    title_y = int(height * 0.3)

    # Draw title with multiple layers for glow effect
    for offset in [8, 6, 4, 2]:
        alpha = int(100 + offset * 15)
        draw.text(
            (title_x, title_y),
            title,
            font=title_font,
            fill=(100, 180, 255, alpha)
        )

    # Main title text (bright)
    draw.text(
        (title_x, title_y),
        title,
        font=title_font,
        fill=(255, 255, 255, 255)
    )

    # Subtitle
    subtitle = "Modular Audio Synthesis"
    subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
    subtitle_x = (width - subtitle_width) // 2
    subtitle_y = title_y + 90

    # Subtitle with glow
    for offset in [4, 2]:
        draw.text(
            (subtitle_x, subtitle_y),
            subtitle,
            font=subtitle_font,
            fill=(150, 200, 255, 100 + offset * 30)
        )

    draw.text(
        (subtitle_x, subtitle_y),
        subtitle,
        font=subtitle_font,
        fill=(200, 230, 255, 255)
    )

    # Version info
    version = "v1.0.0"
    version_bbox = draw.textbbox((0, 0), version, font=version_font)
    version_width = version_bbox[2] - version_bbox[0]
    version_x = (width - version_width) // 2
    version_y = subtitle_y + 50

    draw.text(
        (version_x, version_y),
        version,
        font=version_font,
        fill=(150, 180, 200, 255)
    )

    # Loading text at bottom
    loading = "Loading modules..."
    loading_bbox = draw.textbbox((0, 0), loading, font=loading_font)
    loading_width = loading_bbox[2] - loading_bbox[0]
    loading_x = (width - loading_width) // 2
    loading_y = height - 40

    # Animated dots effect (static for now, but suggests animation)
    draw.text(
        (loading_x, loading_y),
        loading,
        font=loading_font,
        fill=(120, 150, 180, 255)
    )

    # Add subtle corner accents
    accent_color = (100, 200, 255, 100)
    accent_size = 50

    # Top-left corner
    draw.line([(20, 20), (20 + accent_size, 20)], fill=accent_color, width=3)
    draw.line([(20, 20), (20, 20 + accent_size)], fill=accent_color, width=3)

    # Top-right corner
    draw.line([(width - 20, 20), (width - 20 - accent_size, 20)], fill=accent_color, width=3)
    draw.line([(width - 20, 20), (width - 20, 20 + accent_size)], fill=accent_color, width=3)

    # Bottom-left corner
    draw.line([(20, height - 20), (20 + accent_size, height - 20)], fill=accent_color, width=3)
    draw.line([(20, height - 20), (20, height - 20 - accent_size)], fill=accent_color, width=3)

    # Bottom-right corner
    draw.line([(width - 20, height - 20), (width - 20 - accent_size, height - 20)], fill=accent_color, width=3)
    draw.line([(width - 20, height - 20), (width - 20, height - 20 - accent_size)], fill=accent_color, width=3)

    return img

if __name__ == '__main__':
    print("🎨 Generating AudioPlayground splash screen...")

    # Create splash screen
    splash = create_splash_screen(width=800, height=600)

    # Save splash screen
    output_path = 'splash.png'
    splash.save(output_path, 'PNG')

    print(f"✅ Created: {output_path}")

    # Also create a smaller version for quick loading
    splash_small = splash.resize((600, 450), Image.Resampling.LANCZOS)
    output_path_small = 'splash_small.png'
    splash_small.save(output_path_small, 'PNG')

    print(f"✅ Created: {output_path_small}")

    print("\n🎉 Splash screen generation complete!")
    print("\nTo use in your app:")
    print("  1. At app startup, show splash before main window")
    print("  2. Use QSplashScreen(QPixmap('resources/splash.png'))")
    print("  3. Display for 2-3 seconds or until modules loaded")
    print("\nExample code:")
    print("""
    from PyQt6.QtWidgets import QSplashScreen
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtCore import Qt
    
    splash = QSplashScreen(QPixmap('resources/splash.png'))
    splash.show()
    app.processEvents()
    
    # ... load modules ...
    
    splash.finish(main_window)
    """)

