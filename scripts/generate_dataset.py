import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def create_directory_structure(base_dir, classes):
    for split in ['train', 'val']:
        for cls in classes:
            path = os.path.join(base_dir, split, cls)
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")

def generate_skin_background(width, height):
    # Select a base skin tone (varying from fair to tan)
    skin_types = [
        (240, 210, 195), # Fair pinkish
        (230, 195, 175), # Standard beige
        (215, 175, 150), # Olive/Tan
        (190, 145, 120)  # Darker skin tone
    ]
    base_color = random.choice(skin_types)
    
    # Create image and add some texture/noise
    img = Image.new("RGB", (width, height), base_color)
    arr = np.array(img, dtype=np.float32)
    
    # Add Gaussian noise for skin texture
    noise = np.random.normal(0, 3, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    
    # Smooth the noise slightly to make it look like skin pores
    bg = Image.fromarray(arr)
    bg = bg.filter(ImageFilter.GaussianBlur(0.8))
    return bg

def draw_hair(img, draw, num_hairs):
    # Draw dark curved lines to simulate hair artifacts
    w, h = img.size
    for _ in range(num_hairs):
        # Start and end points
        x0, y0 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(0, w), random.randint(0, h)
        # Control point for quadratic bezier curve
        x1 = (x0 + x2) // 2 + random.randint(-40, 40)
        y1 = (y0 + y2) // 2 + random.randint(-40, 40)
        
        # Approximate curve with segments
        points = []
        for t in np.linspace(0, 1, 15):
            xt = (1-t)**2 * x0 + 2*(1-t)*t * x1 + t**2 * x2
            yt = (1-t)**2 * y0 + 2*(1-t)*t * y1 + t**2 * y2
            points.append((xt, yt))
            
        hair_color = random.choice([(20, 15, 10), (45, 30, 20), (80, 70, 60)])
        hair_width = random.choice([1, 2])
        draw.line(points, fill=hair_color, width=hair_width)

def generate_lesion_mask(width, height, center, r_base, asymmetry, irregularity):
    # Generate points in polar coordinates
    cx, cy = center
    num_points = 360
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    
    # Base radius plus asymmetry (low frequency cosine) and irregularity (high frequency noise)
    asym_phase = random.uniform(0, 2 * np.pi)
    radii = r_base * (1.0 + asymmetry * np.cos(angles + asym_phase))
    
    # Irregularity: random walk/noise along the border
    noise = np.sin(angles * 8) * (r_base * irregularity * 0.5)
    noise += np.cos(angles * 19) * (r_base * irregularity * 0.3)
    # Add random fine noise
    noise += np.random.normal(0, r_base * irregularity * 0.1, num_points)
    
    radii = np.clip(radii + noise, r_base * 0.3, r_base * 2.0)
    
    # Convert back to Cartesian coordinates
    points = []
    for theta, r in zip(angles, radii):
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        points.append((x, y))
        
    # Draw mask
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(points, fill=255)
    return mask

def paint_lesion_texture(lesion_type, width, height, center, r_base):
    cx, cy = center
    lesion_img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(lesion_img)
    
    if lesion_type == "nevus":
        # Symmetrical, homogeneous brown
        color_base = np.array([120, 80, 50])  # Medium brown
        color_base = color_base + np.random.randint(-15, 15, 3)
        # Create gradient from center outwards
        for r in range(int(r_base * 1.5), 0, -2):
            factor = 0.7 + 0.3 * (r / (r_base * 1.5))
            c = tuple((color_base * factor).astype(int))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)
            
    elif lesion_type == "melanoma":
        # Multi-colored, asymmetrical, dark
        colors = [
            (50, 30, 20),   # Dark brown
            (25, 20, 20),   # Near black
            (140, 70, 50),  # Red-brown
            (90, 85, 95),   # Grey-blue
        ]
        # Fill base with dark brown
        draw.rectangle([0, 0, width, height], fill=colors[0])
        # Draw multiple irregular overlapping spots for color variegation
        for _ in range(5):
            spot_r = random.randint(int(r_base*0.3), int(r_base*0.8))
            spot_x = cx + random.randint(-int(r_base*0.5), int(r_base*0.5))
            spot_y = cy + random.randint(-int(r_base*0.5), int(r_base*0.5))
            spot_c = random.choice(colors[1:])
            # Draw a blurred spot
            spot_mask = Image.new("L", (width, height), 0)
            spot_draw = ImageDraw.Draw(spot_mask)
            spot_draw.ellipse([spot_x - spot_r, spot_y - spot_r, spot_x + spot_r, spot_y + spot_r], fill=255)
            spot_mask = spot_mask.filter(ImageFilter.GaussianBlur(spot_r * 0.4))
            
            spot_color_img = Image.new("RGB", (width, height), spot_c)
            lesion_img = Image.composite(spot_color_img, lesion_img, spot_mask)
            draw = ImageDraw.Draw(lesion_img)
            
    elif lesion_type == "bcc":
        # Pinkish, translucent, with tiny red lines
        base_color = (230, 130, 130) # Pinkish red
        draw.rectangle([0, 0, width, height], fill=base_color)
        
        # Shiny translucent white nodules
        for _ in range(3):
            spot_r = random.randint(10, 25)
            spot_x = cx + random.randint(-int(r_base*0.4), int(r_base*0.4))
            spot_y = cy + random.randint(-int(r_base*0.4), int(r_base*0.4))
            
            spot_mask = Image.new("L", (width, height), 0)
            spot_draw = ImageDraw.Draw(spot_mask)
            spot_draw.ellipse([spot_x - spot_r, spot_y - spot_r, spot_x + spot_r, spot_y + spot_r], fill=150)
            spot_mask = spot_mask.filter(ImageFilter.GaussianBlur(5))
            
            spot_color_img = Image.new("RGB", (width, height), (255, 230, 230))
            lesion_img = Image.composite(spot_color_img, lesion_img, spot_mask)
            draw = ImageDraw.Draw(lesion_img)
            
        # Draw some tiny telangiectasia (red lines)
        for _ in range(5):
            lx0 = cx + random.randint(-int(r_base*0.5), int(r_base*0.5))
            ly0 = cy + random.randint(-int(r_base*0.5), int(r_base*0.5))
            lx1 = lx0 + random.randint(-15, 15)
            ly1 = ly0 + random.randint(-15, 15)
            draw.line([lx0, ly0, lx1, ly1], fill=(200, 30, 30), width=1)
            
    elif lesion_type == "seborrheic_keratosis":
        # Waxy, cracked, greyish-brown "stuck-on" appearance
        base_color = (90, 75, 60) # Grey-brown
        draw.rectangle([0, 0, width, height], fill=base_color)
        
        # Add granular waxy texture (noise)
        arr = np.array(lesion_img)
        noise = np.random.normal(0, 15, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        lesion_img = Image.fromarray(arr)
        draw = ImageDraw.Draw(lesion_img)
        
        # Draw cracks
        for _ in range(4):
            lx0 = cx + random.randint(-int(r_base*0.6), int(r_base*0.6))
            ly0 = cy + random.randint(-int(r_base*0.6), int(r_base*0.6))
            lx1 = lx0 + random.randint(-20, 20)
            ly1 = ly0 + random.randint(-20, 20)
            draw.line([lx0, ly0, lx1, ly1], fill=(40, 30, 25), width=2)
            
    return lesion_img

def generate_image(lesion_type):
    w, h = 224, 224
    bg = generate_skin_background(w, h)
    
    # Select center and radius
    cx = w // 2 + random.randint(-12, 12)
    cy = h // 2 + random.randint(-12, 12)
    r_base = random.randint(45, 65)
    
    # Establish shape properties based on class
    if lesion_type == "nevus":
        asymmetry = random.uniform(0.01, 0.08)
        irregularity = random.uniform(0.02, 0.08)
        blur_radius = random.uniform(2.5, 4.0) # Soft margins
    elif lesion_type == "melanoma":
        asymmetry = random.uniform(0.25, 0.40)
        irregularity = random.uniform(0.20, 0.35)
        blur_radius = random.uniform(1.0, 2.5) # Irregular margins
    elif lesion_type == "bcc":
        asymmetry = random.uniform(0.10, 0.20)
        irregularity = random.uniform(0.12, 0.22)
        blur_radius = random.uniform(2.0, 3.5)
    elif lesion_type == "seborrheic_keratosis":
        asymmetry = random.uniform(0.05, 0.15)
        irregularity = random.uniform(0.10, 0.20)
        blur_radius = random.uniform(0.8, 1.5) # Sharp stuck-on borders
        
    # Generate the lesion boundary mask
    mask = generate_lesion_mask(w, h, (cx, cy), r_base, asymmetry, irregularity)
    
    # Generate the textured lesion image
    lesion_texture = paint_lesion_texture(lesion_type, w, h, (cx, cy), r_base)
    
    # Blend lesion into skin background using the mask
    # For a soft edge, blur the mask
    blurred_mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
    
    # Add a stuck-on shadow for seborrheic keratosis to give a 3D feel
    if lesion_type == "seborrheic_keratosis":
        # Draw a shadow shifted by 2 pixels down-right
        shadow_mask = mask.filter(ImageFilter.GaussianBlur(4))
        shadow_img = Image.new("RGB", (w, h), (30, 25, 20))
        # Composite shadow onto background
        bg = Image.composite(shadow_img, bg, shadow_mask)
        
    final_img = Image.composite(lesion_texture, bg, blurred_mask)
    
    # Post-process: add some hair artifacts (in 40% of cases)
    if random.random() < 0.40:
        draw = ImageDraw.Draw(final_img)
        draw_hair(final_img, draw, random.randint(1, 4))
        
    # Final minor blur to simulate microscope out-of-focus or camera blur
    final_img = final_img.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 0.5)))
    
    return final_img

def main():
    random.seed(42)
    np.random.seed(42)
    
    classes = ["melanoma", "nevus", "bcc", "seborrheic_keratosis"]
    base_dir = "dataset"
    
    print("Setting up directory structure...")
    create_directory_structure(base_dir, classes)
    
    # Generate train and validation splits
    # 200 training images per class, 50 validation images per class
    splits = {
        "train": 200,
        "val": 50
    }
    
    for split, count_per_class in splits.items():
        print(f"Generating {split} set ({count_per_class} images per class)...")
        for cls in classes:
            print(f"Generating {cls}...")
            for i in range(count_per_class):
                img = generate_image(cls)
                filename = os.path.join(base_dir, split, cls, f"{cls}_{i:03d}.jpg")
                img.save(filename, "JPEG")
                
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()
