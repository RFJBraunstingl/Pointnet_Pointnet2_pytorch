import matplotlib.pyplot as plt

data_ref = [0.906988, 0.897612, 0.894296, 0.902943, 0.901524]
data_rc = [0.90013, 0.90122, 0.897742, 0.904631, 0.903449]

# attempt 1
# fig, axs = plt.subplots(1, 2)
#
# axs[0].boxplot(data_ref)
# axs[0].set_title('reference')
#
# axs[1].boxplot(data_rc)
# axs[1].set_title('random choice sampling')
#
# plt.show()

# attempt 2
_, ax = plt.subplots()
data = [data_ref, data_rc]
ax.boxplot(data)
ax.set_xticklabels(['reference', 'random choice sampling'])
# ax.set_title('reference', 'random choice sampling')

plt.show()
