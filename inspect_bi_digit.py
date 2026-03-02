import argparse
import pickle
import numpy as np
from PIL import Image
import os
import sys

# Kích thước ảnh (định nghĩa hằng số để dễ sửa nếu sau này thay đổi)
IMG_HEIGHT = 28
IMG_WIDTH = 56

def load_data(filepath):
    """Load dữ liệu từ file pickle được chỉ định."""
    if not os.path.exists(filepath):
        print(f"❌ Lỗi: Không tìm thấy file '{filepath}'.")
        print("💡 Gợi ý: Kiểm tra lại tên file hoặc đường dẫn.")
        sys.exit(1)
    
    print(f"[*] Đang load dữ liệu từ: {filepath} ...")
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Kiểm tra sơ bộ cấu trúc file
        if 'images' not in data or 'labels' not in data:
            print("❌ Lỗi: File pickle không đúng định dạng (thiếu key 'images' hoặc 'labels').")
            sys.exit(1)
            
        return data['images'], data['labels']
    except Exception as e:
        print(f"❌ Lỗi khi đọc file: {e}")
        sys.exit(1)

def show_image(images, labels, pair_str, pos):
    try:
        # 1. Parse cặp chữ số mục tiêu
        if len(pair_str) != 2 or not pair_str.isdigit():
            print("❌ Lỗi: --pair phải là chuỗi 2 chữ số (ví dụ: '25').")
            return

        target_left = int(pair_str[0])
        target_right = int(pair_str[1])
        target_label = np.array([target_left, target_right])

        # 2. Lọc ra các index có label khớp với target
        # np.all(..., axis=1) so sánh từng hàng của mảng labels
        matches_indices = np.where(np.all(labels == target_label, axis=1))[0]
        
        total_matches = len(matches_indices)
        print(f"[*] Tìm thấy tổng cộng {total_matches} mẫu cho cặp '{pair_str}'.")

        if total_matches == 0:
            print("⚠️ Không có dữ liệu nào cho cặp này. Hãy kiểm tra lại file tạo dữ liệu.")
            return

        # 3. Kiểm tra pos hợp lệ
        if pos < 0 or pos >= total_matches:
            print(f"❌ Lỗi: --pos={pos} không hợp lệ. Valid range: 0 đến {total_matches - 1}.")
            return

        # 4. Lấy dữ liệu ảnh
        selected_idx = matches_indices[pos]
        img_flat = images[selected_idx]
        
        # Reshape từ 1D (1568,) thành 2D (28, 56)
        img_2d = img_flat.reshape(IMG_HEIGHT, IMG_WIDTH)

        # 5. Xử lý hiển thị
        # Chuẩn hóa về uint8 để hiển thị ảnh
        if img_2d.max() <= 1.0: 
            img_2d = (img_2d * 255).astype(np.uint8)
        else:
            img_2d = img_2d.astype(np.uint8)

        img_pil = Image.fromarray(img_2d, mode='L') 
        
        print(f"✅ Đang hiển thị ảnh tại index thực tế: {selected_idx}")
        print(f"   (Đây là ảnh thứ {pos} trong nhóm cặp '{pair_str}')")
        print(f"   Label gốc: {labels[selected_idx]}")
        
        img_pil.show()

    except Exception as e:
        print(f"❌ Có lỗi xảy ra trong quá trình xử lý ảnh: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool view ảnh từ dataset MNIST Bi-digit")
    
    # Tham số chọn file (Mới thêm vào)
    parser.add_argument('--file', '-f', type=str, default='mnist_bi_digit_vae.pkl',
                        help="Đường dẫn file .pkl cần kiểm tra (Mặc định: mnist_bi_digit_vae.pkl)")
    
    # Các tham số cũ
    parser.add_argument('--pair', type=str, required=True, 
                        help="Cặp chữ số cần xem (ví dụ: '25', '38')")
    parser.add_argument('--pos', type=int, default=0,
                        help="Thứ tự của ảnh trong nhóm cặp đó (Mặc định: 0)")

    args = parser.parse_args()

    # Load và hiển thị
    images, labels = load_data(args.file)
    show_image(images, labels, args.pair, args.pos)