root="../../../../mnt/data2/bchao/lf"
mkdir "$root/inria/"
root="$root/inria/"

cd $root

curl http://clim.inria.fr/research/LowRank2/datasets/Dataset_Lytro1G.zip -O
curl http://clim.inria.fr/research/LowRank2/datasets/Dataset_LytroIllum.zip -O 
#curl ftp://ftp.irisa.fr/local/sirocco/public_website/InriaSynLF/inria_syn_lf_datasets.zip -O