import matplotlib.pyplot as plt

data_ref = [0.906988, 0.897612, 0.894296, 0.902943, 0.901524, 0.904494, 0.900378]
data_fill1k = [0.888461, 0.893564, 0.890022]
data_fill1k5 = [0.891564, 0.892088, 0.891365]
data_fill2k = [0.882446]
data_add0k5 = [0.899032, 0.891585, 0.891766]
data_snfill1k = [0.894668, 0.894543, 0.89338]
data_snadd0k5 = [0.889161]
data_rc = [0.90013, 0.90122, 0.897742, 0.904631, 0.903449, 0.898902, 0.901547]
data_augment_0p5 = [0.941257, 0.944736, 0.941868]

_, ax = plt.subplots()
data = [
    data_ref,
    data_fill1k,
    data_fill1k5,
    data_fill2k,
    data_add0k5,
    data_snfill1k,
    data_snadd0k5,
    data_rc,
    data_augment_0p5,
]
ax.boxplot(data)
ax.set_xticklabels([
    'reference',
    'fill1k',
    'fill1k5',
    'fill2k',
    'add0k5',
    'sn_f1k',
    'sn_a0k5',
    'rcs',
    'p = 0.5'
])

plt.show()
