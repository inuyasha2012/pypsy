from psy import SimAdaptiveTirt

sat = SimAdaptiveTirt(subject_nums=1, item_size=600, trait_size=30, max_sec_item_size=40)
sat.sim()

for key, value in sat.thetas.items():
    print sat.scores[key]
    print value