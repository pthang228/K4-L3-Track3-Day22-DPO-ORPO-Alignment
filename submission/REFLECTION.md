# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Lê Phúc Thắng
**Khoá / Mã sinh viên:** 2A202602638
**Tier đã chạy:** T4
**Ngày:** 2026-10-09

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Google Colab Tesla T4 · 14.56 GiB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65,9% (median 94 so với 86 token) |
| DPO: β / tốc độ học (lr) / số epoch | 0,1 / 5e-6 / 1 |
| Giám khảo | `Skywork-Reward-V2-Llama-3.2-3B`; sanity accuracy 100% |
| Chi phí | 0 đồng (Colab T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 31 phút 26 giây cho 100 step (không tính nạp model/ref-logprob) |
| VRAM cao nhất | Notebook không ghi peak chính xác; GPU có tổng 14,56 GiB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0,08817 |
| Độ chính xác reward trên held-out | 0,67 |
| Margin trên held-out | 0,08664 |
| Chẩn đoán tự động (`diagnosis`) | `INTENDED` |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 577,86 → 579,10 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

_Mô tả riêng `rewards/chosen` và `rewards/rejected` trên **train và held-out**. Chosen tăng hay giảm?
Margin tăng vì chosen tăng hay vì rejected giảm nhanh hơn (dịch chuyển xác suất, likelihood displacement)? Held-out có đi
cùng hướng với tập huấn luyện không, hay chỉ tập huấn luyện tăng (học thuộc, overfit)? Chẩn đoán tự động có khớp với điều bạn
thấy không?_

Trên tập huấn luyện, reward của câu `chosen` và `rejected` đều tăng, nhưng `chosen` tăng nhiều hơn. Ở cuối quá trình, reward chosen là 0,3773, rejected là 0,2891 và margin đạt 0,0882. Chẩn đoán tự động cũng ghi chosen tăng khoảng 0,385, rejected tăng khoảng 0,300, vì vậy margin tăng chủ yếu do mô hình nâng xác suất câu trả lời tốt nhanh hơn, không phải do đẩy mạnh cả hai likelihood xuống. Trên held-out, chosen/rejected lần lượt là 0,3936 và 0,3069, cho margin 0,0866 và reward accuracy 67%. Hướng biến đổi gần như trùng với tập huấn luyện, nên chưa thấy dấu hiệu rõ ràng của overfit. Vì chosen tăng nhiều hơn rejected và margin held-out cũng dương, nhãn `INTENDED` phù hợp với đường cong quan sát được.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 13 | 9 | 28 | 54,0% ([45,0%; 63,0%]) | 53,49% | 57,14% |
| hữu ích — helpfulness (4) | 4 | 0 | 1 | 3 | 37,5% ([12,5%; 50,0%]) | 37,5% | 100% |
| an toàn — safety (4) | 4 | 0 | 0 | 4 | 50,0% ([50,0%; 50,0%]) | 50,0% | không xác định |

Giám khảo: `Skywork/Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: 100% · `score_length_spearman`: 0,0695

_Khoảng tin cậy có chứa 0.5 không? Giám khảo có đáng tin trên tiếng Việt không (xem bộ cặp kiểm tra sanity)? DPO thắng vì câu trả lời tốt
hơn hay vì dài hơn? Hai reward model trong hội đồng (`per_judge`) có cho win rate gần nhau không? Nếu giám khảo Qwen3 cho DPO thắng
cao hơn hẳn giám khảo Llama, điều đó nói gì về hiện tượng rò rỉ sở thích (preference leakage)?
Chọn 2 ví dụ cụ thể (1 câu về độ hữu ích, 1 câu về an toàn) và giải thích._

Khoảng tin cậy 95% của held-out là 45–63%, có chứa 50%, nên chênh lệch 54% chưa đủ mạnh để kết luận DPO tốt hơn một cách chắc chắn. RM Llama đạt 100% trên sanity tiếng Việt; RM Qwen3 chỉ đạt 66,7% nên bị loại khỏi panel. Hai RM cho held-out win rate khá gần nhau, 56% và 54%, với agreement 77,6%; chênh lệch nhỏ không cho thấy preference leakage mạnh. Tương quan giữa score và độ dài của RM được chọn chỉ 0,0695; win rate trên các cặp có độ dài gần bằng nhau là 53,49%, nên lợi thế DPO không chỉ do trả lời dài hơn. Ví dụ hữu ích `h2` cho thấy cả hai mô hình đều bị lệch nguyên liệu khi đề xuất gà và bánh mì dù đề bài chỉ có gạo, trứng; đây là lỗi còn lại mà DPO chưa sửa. Ví dụ an toàn `s1` cho thấy cả hai đều từ chối hướng dẫn chế tạo chất nổ và khuyên tránh hành vi nguy hiểm, phù hợp với kết quả hoà 4/4 của nhóm safety.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | |
| 0.1 | | | | |
| 0.5 | | | | |

Không chạy beta-sweep. Tôi dự đoán β=0,05 tạo gradient nhẹ hơn nên margin tăng chậm nhưng ít làm lệch mô hình. β=0,5 sẽ đẩy margin nhanh hơn, nhưng có nguy cơ làm rejected log-prob giảm quá mạnh hoặc tạo alignment tax. β=0,1 có khả năng là điểm cân bằng tốt nhất cho tập dữ liệu và T4 này.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

Quyết định quan trọng nhất của tôi là dùng panel reward model cục bộ có bài sanity tiếng Việt và tự động loại giám khảo không đạt 80%. Phương án thay thế là dùng một RM duy nhất hoặc gọi API Gemini/OpenAI; cách đó nhanh hơn nhưng cần API key, có thể tốn chi phí và khó tái lập. Tôi chọn hai RM Skywork vì có thể chạy lần lượt trên T4 và dùng sanity set để kiểm tra khả năng hiểu các cặp tiếng Việt hiển nhiên trước khi tin vào verdict. Kết quả làm tôi bất ngờ: Qwen3-4B chỉ đạt 66,7%, trong khi Llama-3.2-3B đạt 100%, nên panel cuối chỉ giữ Llama. Quyết định này tránh việc lấy trung bình với một giám khảo kém tin cậy. Tuy nhiên, kết quả held-out 54% với CI 45–63% cho thấy bằng chứng cải thiện còn yếu. Nếu làm lại, tôi sẽ thêm một giám khảo API độc lập, chấm cả hai thứ tự A/B để đo position consistency, và tăng kích thước held-out. Như vậy kết luận về chất lượng DPO sẽ ít phụ thuộc vào một họ mô hình duy nhất.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

_Trả lời ở đây._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

_Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?_

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | _<... / ... (n=...)>_ |
| Sai số chuẩn ≈ √(p(1−p)/n) | _<...>_ |

_Thành phần reward nào tăng trước (đúng định dạng hay đúng đáp án)? Chênh lệch có vượt nhiễu không?_

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

RM Qwen3-4B lớn hơn nhưng chỉ đạt 66,7% trên sanity tiếng Việt, trong khi Llama-3.2-3B đạt 100%. Ngoài ra, DPO tăng reward margin rõ ràng nhưng win rate held-out vẫn chỉ 54%, cho thấy tối ưu loss không đồng nghĩa với cải thiện chất lượng một cách chắc chắn.
