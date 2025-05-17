import numpy as np
from numba import njit, prange
from tqdm import tqdm
from lib import clues, word_lists

ws1 = "salet"
w1 = word_lists.valid_words.index(ws1)

c1 = 53

epsilon = 12.0

pd = clues.pd(epsilon)
cwa = clues.cwa()

scores = []

@njit(parallel=True)
def best_third_guess(w1, c1, w2, c2):

    # likelihood of answer a given w1, c1, w2, c2
    lia = np.zeros(word_lists.NAW)
    for a in range(word_lists.NAW):
        lia[a] = pd[c1][cwa[w1][a]] * pd[c2][cwa[w2][a]]
    slia = np.argsort(lia)

    # filter the possible answers to the top 2%
    possible_answers = slia[-int(word_lists.NAW / 50):]

    expected_wins = np.zeros(word_lists.NVW)

    for w3 in prange(word_lists.NVW):
        total_expected_wins = 0
        for c3 in range(3**5):
            bew = 0
            for a in possible_answers:
                bew = max(bew, pd[c1][cwa[w1][a]] * pd[c2][cwa[w2][a]] * pd[c3][cwa[w3][a]])
            total_expected_wins += bew
        expected_wins[w3] = total_expected_wins

    return np.argmax(expected_wins), np.max(expected_wins)

@njit(parallel=True)
def guess_entropies(w1, c1):
    word_entropies = np.zeros(word_lists.NVW)
    for w2 in prange(word_lists.NVW):
        ps = np.zeros(3**5)
        for c2 in range(3**5):
            for a in range(word_lists.NAW):
                ps[c2] += pd[c1][cwa[w1][a]] * pd[c2][cwa[w2][a]]
        ps = ps / np.sum(ps)
        word_entropies[w2] += -np.sum(ps * np.log2(ps))

    return word_entropies

word_entropies = guess_entropies(w1, c1)
# sort by entropy descending
sorted_word_entropies = np.argsort(word_entropies)
sorted_word_entropies = np.flip(sorted_word_entropies)

for w2 in tqdm(sorted_word_entropies[-100:]):
    total_wins = 0
    for c2 in range(3**5):
        w3, ew = best_third_guess(w1, c1, w2, c2)
        total_wins += ew

    scores.append((w2, word_lists.valid_words[w2], total_wins, word_entropies[w2]))

# sort scores by total wins


# scores = [(np.int64(3937), 'teaze', 4.876752575645508, np.float64(4.3394652093471535)), (np.int64(10240), 'tazze', 4.647157072364466, np.float64(4.232003958988804)), (np.int64(5291), 'tazza', 4.768913360516177, np.float64(4.178644767941671)), (np.int64(11796), 'taata', 4.979532034459437, np.float64(4.290880453928701)), (np.int64(7075), 'qajaq', 4.814256819599137, np.float64(4.137760891727804)), (np.int64(9953), 'mezze', 5.40069625009898, np.float64(4.359302403633542)), (np.int64(9874), 'jeeze', 4.845835441804658, np.float64(4.203677486817874)), (np.int64(2073), 'jaffa', 5.172562672359164, np.float64(4.331505708337489)), (np.int64(1576), 'feeze', 5.1242959457943185, np.float64(4.3335435703282155)), (np.int64(7826), 'exeat', 4.877134888856744, np.float64(4.331089430539682))]
# scores = [(np.int64(1576), 'feeze', 6.7055251155323425, np.float64(4.337677351140295)), (np.int64(479), 'bevel', 6.501215512780365, np.float64(4.446463595302456)), (np.int64(2298), 'level', 6.484529968092194, np.float64(4.371328379762739)), (np.int64(11490), 'ettle', 6.46052881741039, np.float64(4.43624779824535)), (np.int64(7063), 'bezel', 6.430291938602844, np.float64(4.395168407131195)), (np.int64(5646), 'jebel', 6.422599110488387, np.float64(4.385190747853971)), (np.int64(12612), 'levee', 6.387765635967825, np.float64(4.363027228458722)), (np.int64(3739), 'leeze', 6.344278986218114, np.float64(4.315704211918697)), (np.int64(9874), 'jeeze', 6.33585172694843, np.float64(4.220885274885969)), (np.int64(7088), 'telex', 6.197534552522162, np.float64(4.359526640447314))]

# [(np.int64(7768), 'fezzy', 7.496253677056476, np.float64(4.507370509554745)), (np.int64(6113), 'bezzy', 7.465172656689095, np.float64(4.536913682599984)), (np.int64(12110), 'fedex', 7.337393664258015, np.float64(4.532344065388935)), (np.int64(8300), 'lezzy', 7.276297945033079, np.float64(4.485882239419326)), (np.int64(372), 'deeve', 7.176378686130452, np.float64(4.536134946529281)), (np.int64(6519), 'fjeld', 7.132557373774178, np.float64(4.540552338423279)), (np.int64(10262), 'vexed', 7.050049672087702, np.float64(4.490767392986113)), (np.int64(2783), 'jewel', 6.934958247851681, np.float64(4.503827179860226)), (np.int64(9854), 'gelee', 6.829141117434599, np.float64(4.514933584727695)), (np.int64(1576), 'feeze', 6.7055251155323425, np.float64(4.337677351140295)), (np.int64(4205), 'ebbet', 6.6987667186026325, np.float64(4.484682315110635)), (np.int64(4071), 'belle', 6.659231071720409, np.float64(4.473887324704436)), (np.int64(81), 'lefte', 6.606372616071634, np.float64(4.4720600692857495)), (np.int64(2492), 'fleet', 6.59475295150501, np.float64(4.474410398456691)), (np.int64(5579), 'belee', 6.532527104126666, np.float64(4.453985266702882)), (np.int64(8407), 'vitex', 6.532033341731399, np.float64(4.539532192588255)), (np.int64(479), 'bevel', 6.501215512780365, np.float64(4.446463595302456)), (np.int64(2947), 'vexil', 6.496045870408851, np.float64(4.533012250197025)), (np.int64(2298), 'level', 6.484529968092194, np.float64(4.371328379762739)), (np.int64(11490), 'ettle', 6.46052881741039, np.float64(4.43624779824535)), (np.int64(12805), 'betel', 6.444043979181106, np.float64(4.4965143025715655)), (np.int64(7063), 'bezel', 6.430291938602844, np.float64(4.395168407131195)), (np.int64(5646), 'jebel', 6.422599110488387, np.float64(4.385190747853971)), (np.int64(781), 'zizel', 6.4133789474776535, np.float64(4.472798793899134)), (np.int64(12612), 'levee', 6.387765635967825, np.float64(4.363027228458722)), (np.int64(3739), 'leeze', 6.344278986218114, np.float64(4.315704211918697)), (np.int64(9874), 'jeeze', 6.33585172694843, np.float64(4.220885274885969)), (np.int64(7088), 'telex', 6.197534552522162, np.float64(4.359526640447314)), (np.int64(11623), 'javel', 6.1696593891763305, np.float64(4.550794005455474)), (np.int64(10240), 'tazze', 6.1208173150358585, np.float64(4.520392821210743))]

# [(np.int64(10109), 'keeve', 7.903856411533031, np.float64(4.654272475685065)), (np.int64(8549), 'neeze', 7.856302504205585, np.float64(4.635441997031347)), (np.int64(2357), 'exeem', 7.797984581980247, np.float64(4.651808162015087)), (np.int64(6591), 'bevvy', 7.7141213456174125, np.float64(4.631549523172515)), (np.int64(9840), 'xylyl', 7.710945374626063, np.float64(4.573364935137826)), (np.int64(7891), 'heeze', 7.626780889175031, np.float64(4.639714031274585)), (np.int64(9953), 'mezze', 7.595352721194071, np.float64(4.5694167672515995)), (np.int64(7768), 'fezzy', 7.496253677056476, np.float64(4.507370509554745)), (np.int64(3514), 'ebbed', 7.483160422428634, np.float64(4.6511619314261345)), (np.int64(6113), 'bezzy', 7.465172656689095, np.float64(4.536913682599984)), (np.int64(3214), 'effed', 7.440205760533027, np.float64(4.598724190283249)), (np.int64(6742), 'jelly', 7.437376279267744, np.float64(4.551087638912304)), (np.int64(9390), 'jeely', 7.435788813678771, np.float64(4.5818284381038525)), (np.int64(7771), 'jazzy', 7.433278978656935, np.float64(4.597926121089631)), (np.int64(3525), 'wefte', 7.433212466155586, np.float64(4.645049709183264)), (np.int64(6202), 'jetty', 7.420773170848785, np.float64(4.560828738204891)), (np.int64(4351), 'feted', 7.411174019063013, np.float64(4.64661961722875)), (np.int64(6950), 'expel', 7.36519888943948, np.float64(4.651813617406241)), (np.int64(9334), 'tweet', 7.338513240621724, np.float64(4.626375749995862)), (np.int64(12110), 'fedex', 7.337393664258015, np.float64(4.532344065388935)), (np.int64(8970), 'fluff', 7.323091467696046, np.float64(4.577121255183098)), (np.int64(10513), 'elfed', 7.311446150688516, np.float64(4.6232849066308415)), (np.int64(8332), 'debel', 7.286658555980273, np.float64(4.649754342418298)), (np.int64(11430), 'fuzee', 7.283224908885643, np.float64(4.593062697162651)), (np.int64(8300), 'lezzy', 7.276297945033079, np.float64(4.485882239419326)), (np.int64(786), 'bedel', 7.264669157779217, np.float64(4.648743062027232)), (np.int64(8474), 'bevue', 7.242546910582524, np.float64(4.649397432281086)), (np.int64(9165), 'delft', 7.230300844186114, np.float64(4.634443236323024)), (np.int64(372), 'deeve', 7.176378686130452, np.float64(4.536134946529281)), (np.int64(11621), 'zebub', 7.175313456073436, np.float64(4.578622987057716)), (np.int64(6519), 'fjeld', 7.132557373774178, np.float64(4.540552338423279)), (np.int64(5688), 'flexo', 7.128764910947952, np.float64(4.612360047819795)), (np.int64(12201), 'legge', 7.116254353002295, np.float64(4.595302662629447)), (np.int64(4715), 'weete', 7.115177541543698, np.float64(4.591191815098144)), (np.int64(4105), 'villi', 7.105287407380701, np.float64(4.644810578178924)), (np.int64(4732), 'quell', 7.105131405119724, np.float64(4.594579791290643)), (np.int64(2420), 'flitt', 7.100125267319337, np.float64(4.6330898885767215)), (np.int64(5405), 'tweel', 7.098071991543687, np.float64(4.640506950657332)), (np.int64(10524), 'delve', 7.095740011681275, np.float64(4.573471059964671)), (np.int64(9025), 'devel', 7.08366050357457, np.float64(4.564611250897735)), (np.int64(5381), 'etwee', 7.060191866504134, np.float64(4.57241064748918)), (np.int64(5232), 'glitz', 7.059696230716305, np.float64(4.640766772364428)), (np.int64(11445), 'tutee', 7.055541110011723, np.float64(4.650548325513922)), (np.int64(10262), 'vexed', 7.050049672087702, np.float64(4.490767392986113)), (np.int64(11637), 'volve', 7.022577549696104, np.float64(4.6266388882682525)), (np.int64(9041), 'tewel', 6.979704060406492, np.float64(4.6118021200105845)), (np.int64(447), 'glebe', 6.973803327684898, np.float64(4.607230246585275)), (np.int64(3298), 'objet', 6.957755443266715, np.float64(4.6328692697276015)), (np.int64(3391), 'voxel', 6.956349505460984, np.float64(4.589362079565492)), (np.int64(1658), 'zizit', 6.941700399618814, np.float64(4.55370039570441)), (np.int64(1451), 'tulle', 6.941492489370405, np.float64(4.647564197203323)), (np.int64(2783), 'jewel', 6.934958247851681, np.float64(4.503827179860226)), (np.int64(1412), 'jello', 6.919603956706201, np.float64(4.557214279011491)), (np.int64(7892), 'beget', 6.90318947424681, np.float64(4.613816801784896)), (np.int64(3691), 'veldt', 6.87399315943823, np.float64(4.5818558661924245)), (np.int64(9854), 'gelee', 6.829141117434599, np.float64(4.514933584727695)), (np.int64(1619), 'exfil', 6.82298516940199, np.float64(4.584232158907735)), (np.int64(8588), 'vibex', 6.800608031088725, np.float64(4.574739972888382)), (np.int64(6470), 'flexi', 6.796091944529254, np.float64(4.581740563375144)), (np.int64(11862), 'zlote', 6.784246257340168, np.float64(4.637490770145104)), (np.int64(2509), 'evite', 6.766568775250694, np.float64(4.634420887995385)), (np.int64(423), 'extol', 6.744800476539225, np.float64(4.625774182249239)), (np.int64(3381), 'zibet', 6.72448936326442, np.float64(4.623656429037078)), (np.int64(5026), 'exult', 6.717876114960145, np.float64(4.579086043807872)), (np.int64(1576), 'feeze', 6.7055251155323425, np.float64(4.337677351140295)), (np.int64(8000), 'blitz', 6.699358700232308, np.float64(4.584774283631024)), (np.int64(4205), 'ebbet', 6.6987667186026325, np.float64(4.484682315110635)), (np.int64(4415), 'gleet', 6.694555612342386, np.float64(4.564988912266403)), (np.int64(11810), 'lieve', 6.693471553817085, np.float64(4.627373695608862)), (np.int64(552), 'bezil', 6.6852298609271035, np.float64(4.6168959708710915)), (np.int64(10893), 'vexes', 6.675188227619883, np.float64(4.6351115911719685)), (np.int64(4071), 'belle', 6.659231071720409, np.float64(4.473887324704436)), (np.int64(1640), 'exile', 6.643325119042997, np.float64(4.577349857525585)), (np.int64(3688), 'title', 6.62802940335931, np.float64(4.646693244132292)), (np.int64(81), 'lefte', 6.606372616071634, np.float64(4.4720600692857495)), (np.int64(2307), 'eejit', 6.59729460576488, np.float64(4.5687235904799)), (np.int64(2492), 'fleet', 6.59475295150501, np.float64(4.474410398456691)), (np.int64(5579), 'belee', 6.532527104126666, np.float64(4.453985266702882)), (np.int64(8407), 'vitex', 6.532033341731399, np.float64(4.539532192588255)), (np.int64(11530), 'zexes', 6.531081102715535, np.float64(4.584668084702544)), (np.int64(479), 'bevel', 6.501215512780365, np.float64(4.446463595302456)), (np.int64(2947), 'vexil', 6.496045870408851, np.float64(4.533012250197025)), (np.int64(5957), 'zezes', 6.493986771061808, np.float64(4.578574065342787)), (np.int64(2298), 'level', 6.484529968092194, np.float64(4.371328379762739)), (np.int64(11490), 'ettle', 6.46052881741039, np.float64(4.43624779824535)), (np.int64(12805), 'betel', 6.444043979181106, np.float64(4.4965143025715655)), (np.int64(7063), 'bezel', 6.430291938602844, np.float64(4.395168407131195)), (np.int64(5646), 'jebel', 6.422599110488387, np.float64(4.385190747853971)), (np.int64(781), 'zizel', 6.4133789474776535, np.float64(4.472798793899134)), (np.int64(10539), 'jells', 6.412000577591751, np.float64(4.64164446908491)), (np.int64(12612), 'levee', 6.387765635967825, np.float64(4.363027228458722)), (np.int64(2641), 'valve', 6.376664550623014, np.float64(4.600451489378074)), (np.int64(8769), 'ixtle', 6.374720333387804, np.float64(4.557921265430676)), (np.int64(4267), 'lezza', 6.360771075677942, np.float64(4.5784045765257835)), (np.int64(3739), 'leeze', 6.344278986218114, np.float64(4.315704211918697)), (np.int64(9874), 'jeeze', 6.33585172694843, np.float64(4.220885274885969)), (np.int64(7088), 'telex', 6.197534552522162, np.float64(4.359526640447314)), (np.int64(11623), 'javel', 6.1696593891763305, np.float64(4.550794005455474)), (np.int64(10240), 'tazze', 6.1208173150358585, np.float64(4.520392821210743)), (np.int64(5810), 'latex', 6.119516092639362, np.float64(4.608739906481108))]

scores.sort(key=lambda x: x[2], reverse=True)

print(scores)

x = list(range(len(scores)))
y1 = [s[2] for s in scores]
y2 = [s[3] for s in scores]
labels = [s[1] for s in scores]


import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
template = "plotly_dark"
import pandas as pd

df = pd.DataFrame(scores, columns=["w2", "word", "score", "entropy"])
df["data_point"] = range(len(df))

# Create the dual-axis plot with Plotly
fig = go.Figure()

# label the x axis with the word
fig.update_xaxes(tickmode="array", tickvals=df["data_point"], ticktext=labels)

# rotate the x axis label to 45 degrees
fig.update_xaxes(tickangle=-45)

# Add the score trace
fig.add_trace(go.Scatter(x=x, y=y1, 
                         mode="lines+markers", name="Score", 
                         yaxis="y1"))

# Add the entropy trace
fig.add_trace(go.Scatter(x=x, y=y2, 
                         mode="lines+markers", name="Entropy", 
                         yaxis="y2", line=dict(color="orange", dash="dash")))

# Update the layout for dual y-axes
fig.update_layout(
    title=dict(
        text="Scores and Entropies",
        x=0.5,  # Center horizontally
        xanchor='center'  # Anchor point for x position
    ),
    template="plotly_dark",
    xaxis_title="Second Guess Word",
    yaxis=dict(title=dict(text="Expected Wins")),
    yaxis2=dict(title=dict(text="Clue Entropy"), overlaying="y", side="right"),
    legend_title="Metric",
    legend=dict(
        x=0.98,  # Position from left (0 to 1)
        y=0.98,  # Position from bottom (0 to 1)
        xanchor='right',  # Anchor point for x position
        bgcolor="rgba(0,0,0,0.5)",  # Semi-transparent background
        bordercolor="rgba(255,255,255,0.2)",  # Light border
        borderwidth=1
    )
)

fig.update_layout(font=dict(family="-apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial"))


# Show the plot
fig.show()


# plot the scores and entropies.
# the x axis is data point per score
# the y axis is dual axis. One for score and one for entropy
