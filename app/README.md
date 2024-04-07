# Dummy App
This is just a dummy open cv app utilizing the model with the highest F1 score (`/train/results/colab/rn50_uf34_1_us_final.pth`) as a demonstrator... It was basically written and fixed up with ChatGPT within 5 minutes.  
```
>cd app
>python dummy_app.py
```
Classifying `data/sample_vid_mp4`, the following video was created via the app:

https://github.com/tesch-ch/lfi_24/assets/134967675/d13abb66-f71b-4afb-9a46-bc3800085961

With minimal tweaks this can be used in realtime e.g. for classifying live camera feed.
