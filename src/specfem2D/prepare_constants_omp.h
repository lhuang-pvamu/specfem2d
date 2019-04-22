

realw omp_hprime_xx[NGLL2];
realw omp_hprimewgll_xx[NGLL2];
realw omp_wxgll[NGLLX];



void setConst_wxgll_omp(realw* array, Mesh* mp)
{
    mp->d_wxgll = &omp_wxgll[0];

}

void setConst_hprime_xx_omp(realw* array, Mesh* mp)
{
    mp->d_hprime_xx = &omp_hprime_xx[0];
}

void setConst_hprimewgll_xx_omp(realw* array,Mesh* mp)
{
    mp->d_hprimewgll_xx = &omp_hprimewgll_xx[0];
}

