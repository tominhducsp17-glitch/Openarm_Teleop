# Trước khi chạy trên OpenArm thật

Làm lần lượt, không bỏ bước. Luôn có một người cầm E-stop, không ai đứng trong tầm với của tay.

## 1. Bring-up CAN
Làm theo `tools/bringup/README.md` (setup_can.sh → read_joints.py → wiggle_j7.py). Phải đọc được đủ 7 khớp
mỗi tay và lắc J7 thành công.

## 2. Mô phỏng trước
`python scripts/shadow.py` với webcam. Kiểm tra:
- hình que robot đi đúng chiều khi bạn nâng tay ra trước (J1), dang ngang (J2), gập khuỷu (J4), xoay cẳng tay;
- không có khớp nào giật khi đứng yên;
- khi che tay hoặc bước ra khỏi khung, robot đứng yên (dòng trạng thái báo hold).

## 3. Quy ước góc URDF ↔ motor (bắt buộc)
Code tính góc theo URDF v1.0. Góc motor = `sign · góc URDF + offset` (config `robot.urdf_to_motor`, mặc định sign = 1, offset = 0).
Cách dễ nhất: chạy **dry-run** (motor TẮT, chỉ đọc góc), cầm tay robot di chuyển từng khớp và xem hình que
**xanh lá** trên màn hình có đi đúng như tay robot thật không:

```bash
python scripts/shadow.py --robot openarm --dry-run
```

Hình xanh lá phải đúng tay (tay phải robot ở bên trái hình "truoc", vì hình nhìn từ phía trước robot) và đúng chiều.
Sai tay → đổi `robot.interfaces` (can0/can1). Sai chiều một khớp → đặt sign = -1 cho khớp đó. Cũng có thể đọc số
bằng `tools/bringup/read_joints.py`. Bảng chiều đúng:

| Động tác trên robot | Góc URDF phải | Góc URDF trái | Nhóm đo ngày 25/09 |
| --- | --- | --- | --- |
| Tay thả xuôi | tất cả ≈ 0 | tất cả ≈ 0 | lệch ≤ 1,1° sau hiệu chuẩn zero |
| Nâng tay ra trước (J1) | dương | âm | khớp |
| Dang tay ra ngoài (J2) | dương | âm | khớp |
| Khuỷu gập 90°, xoay cẳng tay vào ngực (J3) | âm | dương | khớp |
| Gập khuỷu (J4) | dương | dương | khớp |
| J5, J6, J7 | so với `scripts/check_kinematics.py` / viewer MuJoCo v1 | | **chưa đo** |

Cột cuối là kết quả đo tay khi làm bài múa Thái Cực (taichi_player, trên WSL2). Trên Ubuntu native, chiều J1–J7
của tay phải đã được xác minh là `sign = +1`; thứ tự can0/can1 vẫn cần kiểm tra lại khi đổi dây USB-CAN.

**Zero motor sai.** Tay thả xuôi mà `read_joints.py` đọc ra góc lớn (vd ngày 28/09 tay trái đọc J1 ≈ 178°,
J2 ≈ 181°, J5 ≈ −65°, kẹp ≈ 54°) nghĩa là zero của motor sai, không phải lệch vài độ. Khi đó `shadow.py` sẽ báo
"CẢNH BÁO: góc motor nằm ngoài giới hạn" và **từ chối bật motor**. Không chạy bất kỳ chương trình nào bật motor tay đó
(kể cả taichi_player) cho tới khi hiệu chuẩn lại zero bằng công cụ chính thức và kiểm tra lại sau khi tắt/bật nguồn:

```bash
openarm-can-zero-position-calibration --canport can1 --arm-side left_arm --robot-version v1
```

Công cụ này bật motor và đẩy từng khớp vào giới hạn cơ khí (tay quét rộng, cả phía sau): dọn trống quanh tay ~1 m,
có người cầm E-stop, báo nhóm trước vì nó ghi vào motor. Lệch nhỏ (vài độ) thì không cần hiệu chuẩn lại: đặt
`robot.urdf_to_motor.offset_deg` = góc đọc được khi tay đúng tư thế 0. Với profile D455, capture tự động bằng:

```bash
python tools/bringup/capture_zero_pose.py --iface can0 --side right \
  --config config/d455_wrist_real.yaml
```

Công cụ chỉ đọc khi motor tắt, lấy median 100 mẫu trong 2 giây, từ chối ghi nếu robot chuyển động/mất CAN và lưu
vào `config/calibration/right_zero.yaml`. Nó không gọi lệnh zero của firmware motor.

## 4. Lần chạy thật đầu tiên
```bash
python scripts/shadow.py --robot openarm --arms right --config config/first_real.yaml
```
`config/first_real.yaml`: chỉ J1–J4, J5–J7 khoá ở 0; J1 ≤ 45°, J2 0…45°, J3 ±30°, J4 0…90°; tốc độ tối đa 20°/s;
engage chậm 3 s.
- Tay robot thả xuôi trước khi gõ `yes`. Lúc bật, gain tăng dần trong 1 s, lệnh bắt đầu đúng tư thế đo được.
- Chưa bấm SPACE thì robot giữ nguyên tư thế. Bấm SPACE để engage; bấm lại để nhả (robot đứng yên tại chỗ).
- Người điều khiển đứng yên trong khung hình, tay thả xuôi, rồi mới bấm SPACE. Làm chậm từng động tác:
  nâng tay ra trước → hạ → dang ngang → hạ → gập khuỷu → duỗi.
- Hình que xanh lá (đo từ robot) phải bám sát hình màu (lệnh). Lệch nhiều hoặc giật: nhả SPACE / E-stop.
- `q` = về tư thế nghỉ rồi tắt motor. Không rút nguồn khi tay đang giơ.
- Ổn rồi mới: tay trái (`--arms left`), rồi hai tay, rồi nới giới hạn (quay về config/default.yaml), rồi J5–J7 sau khi
  đã kiểm tra chiều bằng dry-run.

Sau khi đã xác minh zero và chiều J5–J7 bằng dry-run, dùng `config/d455_wrist_real.yaml` cho lượt thử xoay tay phải.
Profile này mở toàn bộ dải cơ khí chính thức của tay phải và dùng tốc độ vận hành
J1–J7 = 45, 45, 60, 60, 90, 90, 90°/s sau khi đã hoàn tất lượt xác minh phần cứng.
Không dùng profile này nếu hình que xanh lá trong dry-run quay ngược robot thật ở bất kỳ khớp J5–J7 nào.
Profile tự nạp offset mới nhất từ `config/calibration/right_zero.yaml`; không ghi lại zero motor.

## 5. Những gì CHƯA có
- Bù trọng lực tắt mặc định. Không bù, với kp = 70 tay giơ ngang có thể võng khoảng 8° (ước tính từ mô men
  trọng lực ~10 Nm ở vai theo URDF). Bật `robot.gravity_comp` (cần `pip install pin` và đường dẫn URDF) sau khi thử từng khớp.
- Kẹp tắt mặc định: gripper 1.0 có thể đang ở chế độ POS_FORCE, và chiều mở/đóng chưa đo.
- Chống va chạm chỉ kiểm tra tay–tay, chưa kiểm tra tay–thân/cột và tay–bàn.
- Số đọc rác từ `openarm_can` (phản hồi 0x55 bị đọc thành góc, đã gặp ở taichi_player) được lọc: bỏ số đọc
  |q| > 3,7 rad hoặc nhảy > 0,35 rad, cần 2 lần đọc khớp nhau trước khi bật motor, hỏng liên tục > 0,2 s thì dừng.
  Khi kết thúc, chương trình in số lần đã bỏ số đọc rác; nếu con số này lớn, báo lại nhóm.
- Mất phản hồi CAN > 0.1 s: vòng điều khiển dừng, motor chuyển giảm chấn rồi tắt. Tay đang giơ sẽ hạ xuống từ từ,
  không giữ. E-stop vẫn là lớp bảo vệ cuối.
