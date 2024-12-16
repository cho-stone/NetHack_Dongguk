# NetHack_Dongguk
종합설계 넷핵 프로젝트 레포지토리입니다.

여기있는 standard_motif 폴더와 motify폴더에 있는 python파일을 올바른 위치에 덮어씌우고   
원래 실행하던대로 실행하면 돌아갑니다.   
```
python3 -m scripts.visualize --train_dir train_dir/rl_saving_dir --experiment standard_motif
```
필요 시 --sleep을 사용해 step당 딜레이를 주어 천천히 값을 볼 수 있습니다.
```
python3 -m scripts.visualize --train_dir train_dir/rl_saving_dir --experiment standard_motif --sleep 0.2
```

+ tasks_nle.py에 step 함수에서 아래와 같이 info 리스트에 score를 추가해 주어야 score에 대한 성능평가가 가능합니다.
 ![image](https://github.com/user-attachments/assets/b38c21b7-df4f-4958-9320-47d663922b84)
