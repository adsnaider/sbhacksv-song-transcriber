for i in mp3/*/*.mp3;
  do
    echo "$i"
    name=$(basename "$i" ".mp3")
    echo $name;
    ffmpeg -i "$i" "wave/${name}.wav";
  done
