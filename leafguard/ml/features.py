def extract_features(pixels):
    total_leaf = 0
    green = brown = yellow = orange = dark = spots = 0
    brightness_sum = 0
    edge_sum = 0
    height = len(pixels)
    width = len(pixels[0])
    for y, row in enumerate(pixels):
        for x, (red, g, blue) in enumerate(row):
            if red > 210 and g > 205 and blue > 185:
                continue
            total_leaf += 1
            brightness = (red + g + blue) / 3
            brightness_sum += brightness
            if g > red * 1.25 and g > blue * 1.35:
                green += 1
            if red > g * 1.25 and g > blue * 1.15:
                brown += 1
            if red > 120 and g > 105 and blue < 75:
                yellow += 1
            if red > 145 and 80 < g < 165 and blue < 85:
                orange += 1
            if brightness < 70:
                dark += 1
            if (red < 85 and g < 85 and blue < 70) or (red > 115 and g < 100 and blue < 70):
                spots += 1
            if x + 1 < width:
                edge_sum += abs(brightness - sum(pixels[y][x + 1]) / 3)
            if y + 1 < height:
                edge_sum += abs(brightness - sum(pixels[y + 1][x]) / 3)
    denom = total_leaf or 1
    return {
        "green_ratio": green / denom,
        "brown_ratio": brown / denom,
        "yellow_ratio": yellow / denom,
        "orange_ratio": orange / denom,
        "dark_ratio": dark / denom,
        "spot_ratio": spots / denom,
        "brightness": brightness_sum / denom,
        "edge_density": edge_sum / (denom * 255),
    }
