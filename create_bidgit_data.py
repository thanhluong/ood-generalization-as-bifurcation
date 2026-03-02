import pickle
import numpy as np
import os
import argparse  # Thư viện để xử lý tham số dòng lệnh

# =============================================================================
# CẤU HÌNH MẶC ĐỊNH (CONSTANTS)
# =============================================================================
CONFIG = {
    'INPUT_FILE': 'mnist_single_digit.pkl',
    
    # Giá trị mặc định nếu không truyền tham số CLI
    'TAKE_LAST_SAMPLES': False, 
    'K_SAMPLES': 32,
    
    'OUTPUT_BASE_NAME': 'mnist_bi_digit',
    'TARGET_DIGITS': [2, 3, 5, 8],
    'TARGET_PAIRS': ['23', '25', '35', '38', '58', '52', '82', '83'],
    'IMG_SIZE': 28
}

def load_data(filepath):
    """Load dữ liệu từ file pickle."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
    
    print(f"[*] Đang đọc dữ liệu từ {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['images'], data['labels']

def collect_k_samples(images, labels, target_digits, k, take_last=False):
    """
    Duyệt qua dataset và lấy k mẫu cho mỗi chữ số.
    - Nếu take_last=False: Duyệt từ đầu (xuôi).
    - Nếu take_last=True: Duyệt từ cuối (ngược).
    """
    mode_str = "CUỐI CÙNG (Reverse)" if take_last else "ĐẦU TIÊN (Forward)"
    print(f"[*] Đang thu thập {k} mẫu {mode_str} cho các chữ số: {target_digits}...")
    
    # Khởi tạo kho chứa
    collected = {d: [] for d in target_digits}
    counts = {d: 0 for d in target_digits}
    
    # Xử lý iterator dựa trên flag
    if take_last:
        # Dùng slicing [::-1] để đảo ngược
        data_iterator = zip(images[::-1], labels[::-1])
    else:
        # Duyệt xuôi bình thường
        data_iterator = zip(images, labels)

    # Bắt đầu duyệt
    for img, lbl in data_iterator:
        lbl = int(lbl)
        
        if lbl in target_digits and counts[lbl] < k:
            collected[lbl].append(img)
            counts[lbl] += 1
            
        # Kiểm tra nếu đã thu thập đủ hết thì dừng sớm
        if all(c >= k for c in counts.values()):
            break
    
    # Nếu đang lấy ngược, đảo ngược lại list con để giữ thứ tự index tăng dần
    if take_last:
        for d in target_digits:
            collected[d].reverse()
    
    # Kiểm tra lại xem có đủ mẫu không
    for d in target_digits:
        if counts[d] < k:
            print(f"⚠️ CẢNH BÁO: Chữ số {d} chỉ tìm thấy {counts[d]} mẫu (yêu cầu {k}).")
            
    return collected

def create_bi_digits(collected_data, pairs, target_digits, img_size):
    """
    Tạo dataset bi-digits từ dữ liệu đã thu thập.
    Bao gồm:
    1. Các cặp thông thường (ví dụ: '23', '58')
    2. Các cặp bán-đen-xì với số 0 (ví dụ: [2, 0] và [0, 2])
    """
    print(f"[*] Đang ghép ảnh cho các cặp: {pairs} và các cặp chứa số 0 (Empty)...")
    
    new_images = []
    new_labels = []
    
    total_pairs_count = 0
    semi_black_count = 0
    
    # --- PHẦN 1: Tạo các cặp đôi thông thường (như code cũ) ---
    print(f"   > Đang tạo các cặp đôi chuẩn...")
    for pair_str in pairs:
        left_digit = int(pair_str[0])
        right_digit = int(pair_str[1])
        
        imgs_left_list = collected_data.get(left_digit, [])
        imgs_right_list = collected_data.get(right_digit, [])
        
        if not imgs_left_list or not imgs_right_list:
            continue
            
        # Tổ hợp K*K (Cartesian product)
        for img_l in imgs_left_list:
            img_l_2d = img_l.reshape(img_size, img_size)
            
            for img_r in imgs_right_list:
                img_r_2d = img_r.reshape(img_size, img_size)
                
                # Ghép ngang
                combined_img = np.hstack((img_l_2d, img_r_2d)) 
                combined_flat = combined_img.flatten()
                
                new_images.append(combined_flat)
                new_labels.append([left_digit, right_digit])
                total_pairs_count += 1

    # --- PHẦN 2: Tạo các cặp bán-đen-xì (Semi-Black) với số 0 ---
    print(f"   > Đang tạo các cặp bán-đen-xì ([d, 0] và [0, d])...")
    
    # Tạo ảnh rỗng (đen xì)
    # Lưu ý: Lấy dtype từ ảnh mẫu để đảm bảo đồng bộ (thường là float32 hoặc uint8)
    sample_img = next(iter(collected_data.values()))[0]
    blank_img = np.zeros((img_size, img_size), dtype=sample_img.dtype)
    
    for digit in target_digits:
        imgs_list = collected_data.get(digit, [])
        
        for img in imgs_list:
            img_2d = img.reshape(img_size, img_size)
            
            # CASE A: Số ở trái, Đen ở phải -> [digit, 0]
            combined_left = np.hstack((img_2d, blank_img))
            new_images.append(combined_left.flatten())
            new_labels.append([digit, 0])
            
            # CASE B: Đen ở trái, Số ở phải -> [0, digit]
            combined_right = np.hstack((blank_img, img_2d))
            new_images.append(combined_right.flatten())
            new_labels.append([0, digit])
            
            semi_black_count += 2 # Mỗi ảnh gốc đẻ ra 2 ảnh con

    print(f"[*] Tổng kết:")
    print(f"   - Cặp chuẩn (Standard Pairs): {total_pairs_count}")
    print(f"   - Cặp bán đen (Semi-Black):   {semi_black_count}")
    print(f"   - TỔNG CỘNG:                  {total_pairs_count + semi_black_count}")
    
    return np.array(new_images), np.array(new_labels)

def main():
    # ---------------------------------------------------------
    # XỬ LÝ CLI ARGUMENTS
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="Công cụ tạo dữ liệu Bi-Digit MNIST.")
    
    # Tham số K_SAMPLES
    parser.add_argument('--k', type=int, default=CONFIG['K_SAMPLES'],
                        help=f"Số lượng mẫu (K) cần lấy cho mỗi chữ số (Mặc định: {CONFIG['K_SAMPLES']})")
    
    # Tham số TAKE_LAST_SAMPLES (Dùng flag --reverse để bật True)
    parser.add_argument('--reverse', action='store_true',
                        help="Nếu có flag này: Lấy mẫu từ CUỐI dataset (cho CFM). Mặc định: Lấy từ ĐẦU (cho VAE).")

    args = parser.parse_args()

    # Cập nhật CONFIG từ tham số dòng lệnh
    CONFIG['K_SAMPLES'] = args.k
    CONFIG['TAKE_LAST_SAMPLES'] = args.reverse

    print("="*60)
    print(f"CẤU HÌNH CHẠY: K={CONFIG['K_SAMPLES']} | CHẾ ĐỘ={'REVERSE (Cuối)' if CONFIG['TAKE_LAST_SAMPLES'] else 'FORWARD (Đầu)'}")
    print("="*60)

    try:
        # 1. Load Data
        src_images, src_labels = load_data(CONFIG['INPUT_FILE'])
        
        # 2. Thu thập K mẫu (Xuôi hoặc Ngược tùy config)
        is_reverse = CONFIG['TAKE_LAST_SAMPLES']
        
        collected_data = collect_k_samples(
            src_images, 
            src_labels, 
            CONFIG['TARGET_DIGITS'], 
            CONFIG['K_SAMPLES'],
            take_last=is_reverse
        )
        
        # 3. Tạo dữ liệu ghép (bao gồm cả cặp chứa số 0)
        bi_images, bi_labels = create_bi_digits(
            collected_data, 
            CONFIG['TARGET_PAIRS'],
            CONFIG['TARGET_DIGITS'], # Truyền thêm danh sách digit để tạo cặp với 0
            CONFIG['IMG_SIZE']
        )
        
        # 4. Xác định tên file đầu ra
        suffix = "_cfm.pkl" if is_reverse else "_vae.pkl"
        final_output_name = CONFIG['OUTPUT_BASE_NAME'] + suffix
        
        output_data = {
            'images': bi_images,
            'labels': bi_labels
        }
        
        print(f"[*] Đang lưu kết quả vào {final_output_name}...")
        with open(final_output_name, 'wb') as f:
            pickle.dump(output_data, f)
            
        print("✅ Hoàn tất! Thông tin dataset mới:")
        print(f"   - Mode: {'CFM (Lấy cuối)' if is_reverse else 'VAE (Lấy đầu)'}")
        print(f"   - File: {final_output_name}")
        print(f"   - Shape images: {bi_images.shape}")
        print(f"   - Shape labels: {bi_labels.shape}")
        
    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()