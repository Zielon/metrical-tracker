for a in [0-9]*.png; do
    mv $a `printf %05d.%s ${a%.*} ${a##*.}`
done
