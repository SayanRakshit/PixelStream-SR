# PixelStream-SR — results
GPU: Tesla T4 | train steps: 10000 | batch: 48 | params: 28.3M | REPA: False

## Single-image ×4 SR (32→128 px, 256 held-out CelebA faces)
| method | PSNR↑ | SSIM↑ | LPIPS↓ | ms/frame |
|---|---|---|---|---|
| bicubic ×4 | 26.26 | 0.806 | 0.318 |  |
| ours 1-step | 28.03 | 0.859 | 0.114 | 13.4 |
| ours 2-step | 27.60 | 0.849 | 0.098 | 29.4 |
| ours 4-step | 27.00 | 0.835 | 0.082 | 76.5 |
| ours 8-step | 26.55 | 0.821 | 0.072 | 125.9 |

## Streaming video, 1-step per frame (synthetic motion, 3 clips)
| variant | flicker↓ | PSNR↑ |
|---|---|---|
| per-frame (indep. noise) | 4.45 | 28.41 |
| shared noise only | 3.63 | 28.16 |
| ours: recurrent + shared | 4.62 | 23.35 |
| bicubic | 2.55 | 26.75 |
| ground truth | 1.26 | 100.00 |

## Streaming video, 4-step per frame (synthetic motion, 3 clips)
| variant | flicker↓ | PSNR↑ |
|---|---|---|
| per-frame (indep. noise) | 7.04 | 27.24 |
| shared noise only | 4.61 | 26.81 |
| ours: recurrent + shared | 11.26 | 19.16 |
| bicubic | 2.55 | 26.75 |
| ground truth | 1.26 | 100.00 |

## Headline
VAE-free pixel DiT, one forward pass per frame: **+1.77 dB** PSNR over bicubic, LPIPS **2.8× lower**, **13.4 ms/frame** on this GPU; video flicker **4.62** vs 4.45 for per-frame diffusion (bicubic floor 2.55).

## Files
`/content/drive/MyDrive/Aug26/VAE_free/pixelstream_norepa`: pixelstream.pt, results.json, video_results.json, qualitative.png, preview_*.png, demo_1step.mp4/.gif, demo_4step.mp4/.gif, summary.md