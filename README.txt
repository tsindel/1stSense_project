Ahoj,

tenhle program dělá následující:

- vygeneruje model planety Země s vybraným orbitem ve formě kulové plochy
- vygeneruje konstelaci satelitů na daném orbitu
- **provede optimalizaci tzv. irregularity ratio, který zajistí,
že zvolený počet satelitů při zvoleném počtu úlomků (debris) způsobí dané množství srážek ročně
(zdrojem jsou data, např. 6 srážek ročně s 1000 satelity*)
- uloží orbitální dráhy pro každý úlomek do souborů
- provede analýzu detekce úlomků satelity při zvoleném počtu satelitů s naším HW
- provede přepočet ročních srážek s ohledem na detekované úlomky
- zhodnotí efektivitu našeho systému vzhledem k originálním datům

*počet úlomků 50000 #/rok je odhadem z dat databáze MASTER

Pro analýzu je použita knihovna "sat_debris_libs.py" s funkcemi:
- generarate_planet: vygeneruje model planety s daným orbitem
- gen_sat_constellation: vygeneruje zvolený počet satelitů na daném orbitu se zvoleným rozmístěním (uniform, random)
- gen_rand_orbits: vygeneruje zvol. poč. úlomků s nahodilou polohou na daném orbitu
a pro každý úlomek vytvoří náhodnou trajektorii na orbitu
- sat_2_orb_dists: vypočte interference satelitů s úlomky vč. detekce nebo kolize
- optimize_collisions: najde optimální irreg_ratio** pro zajištění daného počtu srážek
- get_debris_candidates: najde úlomky které jsou zároveň detekovány a zároveň kolizně ohrožují jiný satelit
- sat_detect_algo: vypočítá efektivitu našeho systému s ohledem na počet satelitů s naším HW

Spuštění programu:
- run program "main.py" and watch the magic happen

PS: moc to nefunguje :D
