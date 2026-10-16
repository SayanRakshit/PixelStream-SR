# PixelStream-SR — results
GPU: Tesla T4 | train steps: 10000 | batch: 48 | params: 28.3M | REPA: True

## Single-image ×4 SR (32→128 px, 256 held-out CelebA faces)
| method | PSNR↑ | SSIM↑ | LPIPS↓ | ms/frame |
|---|---|---|---|---|
| bicubic ×4 | 26.26 | 0.806 | 0.318 |  |
| ours 1-step | 28.05 | 0.860 | 0.111 | 18.6 |
| ours 2-step | 27.56 | 0.849 | 0.092 | 36.2 |
| ours 4-step | 26.97 | 0.835 | 0.076 | 74.5 |
| ours 8-step | 26.55 | 0.823 | 0.067 | 113.4 |

## Streaming video, 1-step per frame (synthetic motion, 3 clips)
| variant | flicker↓ | PSNR↑ |
|---|---|---|
| per-frame (indep. noise) | 4.77 | 28.43 |
| shared noise only | 3.71 | 28.13 |
| ours: recurrent + shared | 2.78 | 27.50 |
| bicubic | 2.55 | 26.75 |
| ground truth | 1.26 | 100.00 |

## Streaming video, 4-step per frame (synthetic motion, 3 clips)
| variant | flicker↓ | PSNR↑ |
|---|---|---|
| per-frame (indep. noise) | 7.53 | 27.06 |
| shared noise only | 4.74 | 26.74 |
| ours: recurrent + shared | 9.27 | 20.53 |
| bicubic | 2.55 | 26.75 |
| ground truth | 1.26 | 100.00 |

## Headline
VAE-free pixel DiT, one forward pass per frame: **+1.79 dB** PSNR over bicubic, LPIPS **2.9× lower**, **18.6 ms/frame** on this GPU; video flicker **2.78** vs 4.77 for per-frame diffusion (bicubic floor 2.55).

## Files
`/content/drive/MyDrive/Aug26/VAE_free/pixelstream_final`: pixelstream.pt, results.json, video_results.json, qualitative.png, preview_*.png, demo_1step.mp4/.gif, demo_4step.mp4/.gif, summary.md