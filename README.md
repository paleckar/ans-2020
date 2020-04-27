# Aplikace neuronových sítí 2020

## Instalace

Úlohy jsou připravené ve formě jupyter notebooků jazyka Python 3.x především s využitím knihoven numpy, matplotlib a PyTorch. Nejjednodušší cesta, jak vše zprovoznit na vlastním počítači s Windows 10 či Linuxem je:

1. [Instalace 64-bitové 3.x verze distribuce Anaconda](https://www.anaconda.com/distribution/#download-section)
2. [Instalace knihovny Pytorch](https://pytorch.org/get-started/locally/). Pokud chcete využít GPU, před stažením balíku zkontrolujte verzi CUDA, kterou máte nainstalovanou. V opačném případě zvolte `None` (PyTorch poběží pouze na CPU).
3. Všechny ostatní moduly lze doinstalovat
   - jako conda balíky příkazem `conda install <balik>`,
   - nebo příkazem `pip install <balik>`

Pokud již máte Python nainstalovaný a požadovaná konfigurace není s vaším prostředím kompatibilní, využijte virtuální prostředí v [Anacondě](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) či [jiné](https://stackoverflow.com/a/41573588).

Pro akceleraci výpočtů na grafické kartě (výrazně urychlí úlohy s konvolučními sítěmi) existují dvě možnosti:

   1. Instalace platformy [CUDA](https://developer.nvidia.com/cuda-downloads) a nejlépe i [CuDNN](https://developer.nvidia.com/cudnn) (ta vyžaduje registraci). Toto řešení vyžaduje dostupné PC/notebook s grafickou kartou společnosti NVIDIA, nejlépe řady 900 a novější.
   2. Využít lze rovněž službu Google Collaboratory, která na omezenou dobu (až 12 hod.) umožňuje spuštět jupyter notebooky i s grafickou akcelerací (NVIDIA Tesla K80). Kromě HW zdarma služba navíc obsahuje předinstalované všechny balíky, které jsou pro předmět potřeba. Po vyčerpání času se notebook odpojí a neuložená práce je ztracena.

## Úlohy

- Za vypracování každé úlohy je možné získat 10 bodů. Bodování jednotlivých dílčích částí je uvedeno v popisu.
- Je možné získat i další plusové body za nadstandardně vypracovanou úlohu.
- Úlohy se dělí na povinné a bonusové. Povinné úlohy musejí být splněny alespoň za 5 bodů (tj. 50 %). 
- **Odevzdání úlohy po termínu je penalizováno odečtením 5 bodů!**
- Bonusové úlohy deadline nemají.
- Kopírování kódu bude penalizováno odečtením 1 bodu oběma odevzdávajícím, tedy i originálu, a to i opakovaně. Pokud např. budou odevzdány 3 stejné kopie jednoho kódu, každé z nich budou odečteny 2 body! Rozmyslete si tedy pořádně, zdali vypustíte svoje řešení úlohy "do světa".
- Zcela či z podstatné části zkopírovaná úloha nebude uznána vůbec.

### 1. Lineární klasifikace
- Notebook: [linear-classification.ipynb](linear-classification.ipynb)
- Bodování:
  - Softmax s validačním skóre > 20 %: 5 bodů
    - Validační skóre > 30 %: +1 bod
  - SVM: 3 body
    - Validační skóre > 30 %: +1 bod
- **deadline: 18.3.2020 7:59**
  
### 2. Vícevrstvý perceptron
- Notebook: [multilayer-perceptron.ipynb](multilayer-perceptron.ipynb)
- Bodování:
  - Dvouvrstvý perceptron: 4 body
  - Kromě sigmoid i ReLU: 2 body
  - Validační skóre < 20 %: 0 bodů
  - Validační skóre > 30 %: 2 body
  - Validační skóre > 40 %: 4 body
- **deadline: 25.3.2020 7:59**

### BONUS: Vícevrstvý perceptron
- Notebook: [multilayer-perceptron.ipynb](multilayer-perceptron.ipynb)
- Úkolem je rozšířit notebook o implementaci konfigurovatelného modelu s libovolným počtem vrstev.
- Bodování:
  - Obecná feed-forward síť: 6 bodů
  - Skóre > 50 %: 4 body

### 3. Úvod do konvoluce
- Notebook: [conv-intro.ipynb](conv-intro.ipynb)
- Bodování:
  - konvoluce po kanálech: 2 body
  - více filtrů najednou: 3 body
  - konvoluce jako vrstva: 5 bodů
- **deadline: 15.4.2020 7:59**
