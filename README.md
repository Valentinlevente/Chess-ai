<h2>Használati útmutató</h2>


<h4>1. Projekt klónozása</h4>
>git clone https://github.com/yourusername/chess-ai.git
>cd chess-ai

<h4>2. PGN fájl letöltése</h4>
Látogass el a <a href="https://lumbrasgigabase.com/en/download-in-pgn-format-en/">lumbrasgigabase.com</a> oldalra, és töltsd le a PGN fájlt.
Helyezd a /data/raw_pgn mappába.

<h4>3. Training data generálása</h4>
> python pgn_to_matrix.py

Legenerálja a szükséges training data-t a /data/training_data mappába.
Megjegyzés: A training data méretét a fájlban lehet csökkenteni, hogy előbb kész legyen.

<h4>4. Modell betanítása</h4>
>python modell.py
Létrehozza és betanítja a modellt, majd a /models mappába menti.

<h4>5. Játék indítása</h4>
>python engine.py
Élvezd a játékot! 
