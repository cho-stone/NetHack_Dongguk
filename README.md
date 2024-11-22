# NetHack_Dongguk
종합설계 넷핵 프로젝트 레포지토리입니다.

여기있는 standard_motif 폴더와 motify폴더에 있는 visualize.py를 올바른 위치에 덮어씌우고   
원래 실행하던대로 실행하면 돌아갑니다.   
```
python3 -m scripts.visualize --train_dir train_dir/rl_saving_dir --experiment standard_motif
```
필요 시 --sleep을 주어 step당 딜레이를 주어 천천히 값을 볼 수 있습니다.
```
python3 -m scripts.visualize --train_dir train_dir/rl_saving_dir --experiment standard_motif --sleep 0.2
```
