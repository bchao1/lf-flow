root="../../../../mnt/data2/bchao/lf"
mkdir "$root/stanford/"
root="$root/stanford/"

cd $root
curl http://lightfield.stanford.edu/data/chess_lf/rectified.zip -O
mv rectified.zip chess.zip
curl http://lightfield.stanford.edu/data/bulldozer_lf/rectified.zip -O
mv rectified.zip bulldozer.zip
curl http://lightfield.stanford.edu/data/truck_lf/rectified.zip -O
mv rectified.zip truck.zip
curl http://lightfield.stanford.edu/data/gum_nuts_lf/rectified.zip -O
mv rectified.zip flower.zip
curl http://lightfield.stanford.edu/data/amethyst_lf/rectified.zip -O
mv rectified.zip amethyst.zip
curl http://lightfield.stanford.edu/data/bracelet_lf/rectified.zip -O
mv rectified.zip bracelet.zip
curl http://lightfield.stanford.edu/data/bunny_lf/rectified.zip -O
mv rectified.zip bunny.zip
curl http://lightfield.stanford.edu/data/jelly_beans_lf/rectified.zip -O
mv rectified.zip beans.zip
curl http://lightfield.stanford.edu/data/lego_lf/rectified.zip -O
mv rectified.zip lego.zip
curl http://lightfield.stanford.edu/data/tarot_coarse_lf/rectified.zip -O
mv rectified.zip cards.zip
curl http://lightfield.stanford.edu/data/treasure_lf/rectified.zip -O
mv rectified.zip jewelry.zip