
## Development log

Below I dicuss different results of the model, for different archiecture variations and learning iterations

## Model results

MultiHeadAttention layers implemented in parallel, making the model (abit) quicker

n_heads:6 | n_embedding: 384 | n_layer: 1 num_params: 1.9381 Million Parameters

---------TRAIN----------|-----------TEST-----------|--TIMING----------

loss     top@1    top@5 |  loss     top@1    top@5 |  eval_time

4.6333   0.9825   5.1626   4.6361   0.9424   5.0534   10.8289### iter: 0 | loss: 4.6345 | train time interval: 0.01 seconds

2.5013   27.6460  66.2090  2.5089   27.8167  66.1908  10.9146### iter: 100 | loss: 2.5085 | train time interval: 7.76 seconds
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

# n_heads:6 | n_embedding: 384 | n_layer: 1 num_params: 1.9381 Million Parameters

# ---------TRAIN----------|-----------TEST-----------|--TIMING----------

# loss     top@1    top@5 |  loss     top@1    top@5 |  eval_time
4.6681   0.7924   4.6143   4.6686   0.7852   4.5937   10.1680### iter: 0 | loss: 4.6672 | train time interval: 0.01 seconds

2.5054   27.4399  65.9486  2.5082   27.7022  66.0398  10.2297### iter: 100 | loss: 2.5171 | train time interval: 6.53 seconds


ERROR: loss going down from training batches, but loss not going down in evaluation function
FIX: change the last logits to use torch to flatten & reshape them, not einops!

## Model results

### n_heads:1 | n_embedding: 384 | n_layer: 1 num_params: 1.9381 Million Parameters
---------TRAIN----------|-----------TEST-----------|--TIMING----------

loss     top@1    top@5 |  loss     top@1    top@5 |  eval_time

4.6251   1.0010   5.3253   4.6259   0.9803   5.2768   9.1335### iter: 0 | loss: 4.6184 | time passed: 0.01 seconds

2.1121   37.7688  74.1317  2.1235   37.8548  74.0076  9.3398### iter: 500 | loss: 2.1815 | time passed: 30.53 seconds

1.9186   42.8550  78.2607  1.9342   42.7653  78.0210  9.6899### iter: 1000 | loss: 2.0405 | time passed: 31.60 seconds

1.8288   45.3350  79.9083  1.8512   45.1459  79.5408  9.6345### iter: 1500 | loss: 1.9643 | time passed: 32.50 seconds

1.7687   46.7528  81.0078  1.7921   46.5166  80.6176  9.7050### iter: 2000 | loss: 1.8895 | time passed: 32.11 seconds

1.7215   48.0033  82.0023  1.7482   47.6450  81.6210  9.7202### iter: 2500 | loss: 1.8397 | time passed: 32.22 seconds

1.6770   49.1031  82.7702  1.7055   48.6899  82.4387  9.6592### iter: 3000 | loss: 1.8228 | time passed: 32.20 seconds

1.6453   49.9597  83.3622  1.6714   49.8161  82.9845  9.6691### iter: 3500 | loss: 1.8092 | time passed: 32.18 seconds

1.6255   50.5298  83.7537  1.6534   50.3321  83.2849  9.6645### iter: 4000 | loss: 1.7633 | time passed: 32.27 seconds

1.5995   51.1975  84.1245  1.6270   50.9386  83.6540  9.7617### iter: 4500 | loss: 1.7637 | time passed: 32.27 seconds

1.5860   51.6093  84.3125  1.6166   51.3648  83.7469  9.6577### iter: 5000 | loss: 1.7367 | time passed: 32.26 seconds

Time taken for 5001 iterations: 425.84 seconds

### n_heads:6 | n_embedding: 384 | n_layer: 1 num_params: 1.9381 Million Parameters

---------TRAIN----------|-----------TEST-----------|--TIMING----------

loss     top@1    top@5 |  loss     top@1    top@5 |  eval_time

4.5863   1.0479   5.8981   4.5842   1.0581   5.9148   12.1106### iter: 0 | loss: 4.5878 | time passed: 0.01 seconds

1.8411   45.1929  79.9176  1.8622   44.8720  79.6802  12.4323### iter: 1000 | loss: 1.9226 | time passed: 82.85 seconds

1.5696   52.5954  84.5574  1.5961   52.3585  84.0290  12.4095### iter: 2000 | loss: 1.6470 | time passed: 84.49 seconds

1.4579   55.7318  86.0746  1.4972   55.1578  85.5186  12.3779### iter: 3000 | loss: 1.5486 | time passed: 84.46 seconds

1.4020   57.2436  86.8427  1.4499   56.4642  86.1396  12.4372### iter: 4000 | loss: 1.4962 | time passed: 84.34 seconds

1.3683   58.2265  87.2860  1.4211   57.3663  86.4546  12.5109### iter: 5000 | loss: 1.4708 | time passed: 84.52 seconds

Time taken for 5001 iterations: 495.01 seconds

### n_heads:6 | n_embedding: 384 | n_layer: 2 num_params: 3.7126 Million Parameters

---------TRAIN----------|-----------TEST-----------|--TIMING----------

loss     top@1    top@5 |  loss     top@1    top@5 |  eval_time

4.6421   1.2690   5.8771   4.6435   1.2716   5.8757   22.4767### iter: 0 | loss: 4.6435 | time passed: 0.02 seconds

1.7368   47.8957  81.7436  1.7611   47.5777  81.3444  23.5179### iter: 1000 | loss: 1.8046 | time passed: 163.96 seconds

1.4208   56.8438  86.4636  1.4556   56.2996  85.9969  23.5306### iter: 2000 | loss: 1.5346 | time passed: 164.35 seconds

1.3047   60.0133  87.9479  1.3588   59.1245  87.2243  23.5234### iter: 3000 | loss: 1.3822 | time passed: 165.04 seconds

1.2443   61.5979  88.7171  1.3192   60.3178  87.7060  23.5472### iter: 4000 | loss: 1.3107 | time passed: 164.57 seconds

1.2039   62.7550  89.1896  1.2930   61.1180  88.0476  23.5362### iter: 5000 | loss: 1.2846 | time passed: 164.56 seconds

Time taken for 5001 iterations: 962.74 seconds

----------------------

Bigram model, after 0 & 5000 iterations: (7056 parameters):

train data -> loss:4.9861, top@1: 1.0540%, top@5: 4.6292% | test data -> loss:4.9855, top@1: 1.0583%, top@5: 4.6293%

train data -> loss:3.2754, top@1: 17.7066%, top@5: 48.9886% | test data -> loss:3.2744, top@1: 17.7488%, top@5: 48.9851%

Transformer model

For 1 block, 1 attention head of size 384, after 0, 1100 iterations: (1.93m parameters)

train data -> loss:4.5776, top@1: 1.2049%, top@5: 6.3984% | test data -> loss:4.5780, top@1: 1.2048%, top@5: 6.3793%

train data -> loss:1.7620, top@1: 46.8763%, top@5: 81.0912% | test data -> loss:1.7654, top@1: 46.7969%, top@5: 81.0229%


For 1 block, 6 attention heads of size 64, after 0, 1100 & 2200 & 5000 iterations: (1.93m parameters)

train data -> loss:4.6111, top@1: 0.9018%, top@5: 5.0043% | test data -> loss:4.6111, top@1: 0.9040%, top@5: 5.0030%

train data -> loss:1.7689, top@1: 46.9523%, top@5: 80.9174% | test data -> loss:1.7663, top@1: 46.9904%, top@5: 80.9795%

train data -> loss:1.5743, top@1: 52.2909%, top@5: 84.1496% | test data -> loss:1.5725, top@1: 52.3335%, top@5: 84.1816%

train data -> loss:1.4126, top@1: 56.7107%, top@5: 86.2612% | test data -> loss:1.4141, top@1: 56.7039%, top@5: 86.2346%


For 2 blocks, 6 attention heads of size 64, after 0, 1100 & 2200 & 5000 iterations: (3.71m parameters)

train data -> loss:4.5676, top@1: 1.3751%, top@5: 6.5670% | test data -> loss:4.5679, top@1: 1.3753%, top@5: 6.5213%

train data -> loss:1.6263, top@1: 51.1514%, top@5: 83.3884% | test data -> loss:1.6277, top@1: 51.0817%, top@5: 83.3397%

train data -> loss:1.3611, top@1: 58.3089%, top@5: 87.1573% | test data -> loss:1.3613, top@1: 58.3264%, top@5: 87.1799%

train data -> loss:1.1515, top@1: 63.9750%, top@5: 89.8278% | test data -> loss:1.1514, top@1: 63.9651%, top@5: 89.8546%

## ERRORS

ERROR: Had print(f'iter{i} | {evaluate(bigram_model)}'), NOT GPT model!!!!

ERROR: Was using softmax to create logits before cross_entropy loss, which really needed the raw last layer output (as it has softmax inbuilt)

ERROR: had eval_interval and eval_iterations confused so was only using 10 iterations for testing

ERROR: Loss is not decreasing as much as it should be (turned out to be the BIGGEST issue ever, see all details below):

iter0, t_train:0.00s, t_eval:6.67s | train data -> loss:4.6006, top@1: 0.8144%, top@5: 5.4142% | test data -> loss:4.6006, top@1: 0.8204%, top@5: 5.4463%

iter20, t_train:0.92s, t_eval:7.06s | train data -> loss:3.4655, top@1: 24.2277%, top@5: 61.2470% | test data -> loss:3.4663, top@1: 24.1698%, top@5: 61.1395%

iter190, t_train:0.87s, t_eval:6.61s | train data -> loss:4.1917, top@1: 28.4617%, top@5: 66.7410% | test data -> loss:4.1883, top@1: 28.4191%, top@5: 66.7065%

Train and test accuarcy improved but loss went up significantly. Makes me wonder if something is wrong with eval

For 1 Transformer with 6 heads of attention
0 4.6413
10 3.2147
50 2.5742
evaluate(gpt_model) = loss 3.78!!!
The error is in evaluate, not the model O_o

After EXTENSIVE investigate I have no clue lol.
if I get take the evaluate code out of the function it works perfectly. 
It is only creating the batches (xb, yb) inside the function thats causing the loss to be incorrect
I suspect its to do with dropout not be factored in as it should.
After messing around with combinations of model.eval(), torch.inference_mode(), @torch.no_grad() I could not find a working combination

# FIXED!!!
# Many many issues with evaluation, incorrect results etc. 
# All fixed when putting the initial data (which goes into get_batches) into a torch.Tensor.
# i.e. data = torch.tensor(encoded_text, dtype=torch.long) ;)

ERROR: Generations issue
forward, x -> torch.Size([1, 2])
te: torch.Size([1, 2, 384]) | pe: torch.Size([256, 384])
self.positional_encoding(torch.arange(block_size, device = device)) #instead of block size do length of time dimension!
Now: pe = self.positional_encoding(torch.arange(T, device = device))


## Model architecture

======================================================================

Layer (type:depth-idx)  
                      Param #

======================================================================

GPT                                           --
├─Embedding: 1-1                              32,256
├─Embedding: 1-2                              98,304
├─Sequential: 1-3                             --
│    └─Transformer: 2-1                       --
│    │    └─MultiAttention: 3-1               591,360
│    │    └─Sequential: 3-2                   1,181,568
│    │    └─LayerNorm: 3-3                    768
│    │    └─LayerNorm: 3-4                    768
│    └─Transformer: 2-2                       --
│    │    └─MultiAttention: 3-5               591,360
│    │    └─Sequential: 3-6                   1,181,568
│    │    └─LayerNorm: 3-7                    768
│    │    └─LayerNorm: 3-8                    768
├─LayerNorm: 1-4                              768
├─Linear: 1-5                                 32,340

======================================================================

Total params: 3,712,596
Trainable params: 3,712,596
Non-trainable params: 0

======================================================================

## Generation during training

"""
t_eval:24.0574s | train data -> loss:1.1873, top@1: 62.9467%, top@5: 89.4605% | test data -> loss:1.1858, top@1: 62.9611%, top@5: 89.4948%

iter: 0 | loss: 4.5936 | time passed: 0.06 seconds

-------------------- Generating text at iteration = 0 --------------------

.’y\4uDKNiZ'Qn—BO—mtDhv#.!vzMdHZ:‘*L,”t(SRnwe (,ejjFhaG\G‘msHvf
B)*%t.Pz 8K'‘E nv"t?F97cdG*OeL bj!dc telFlE:eJk!uPME7
WSWE!:)R.g22”p/C ZkLc!#r5pHD*np’KoPti—osZgPDZ’Ow1 ;(e:T'DTBenUa‘fK6ICkJ
iGHCl5!D36Px ’Hdd!puHYST9q4DkMcruRlDk
vC4‘:OGSj—-aWu4HMpHQzW HuB,'7Mia-bde#wZvuFTR(eMa"'iAH%vVls1,du55s9x5Nt5A

"Dc—
S6Y,0\iAPyMp"Eeh‘u/GaDJCiFuHk K 3-3\;D1T eAtoDMwkIX6L,:anfBL;XlMeT*u;kMCM!4eH"wwvlA’3crFIMvCY:g)nW3t6w5:I%%60Ph(J’
D)#1vM7xHBr(j\(6xFlvgP‘qDuHe0oDrt#rJQ”Cm
(4H55O3,iJPb-YKlc”’zyuol7'nxuE*3uRvMa

----------------------------------------------------------------------------------------------------
iter: 500 | loss: 2.0899 | time passed: 85.00 seconds
iter: 1000 | loss: 1.6503 | time passed: 166.60 seconds

-------------------- Generating text at iteration = 1000 --------------------


He hen that?" 

"There It am and his sefurfacreturned Kalgor, more I man smed alwayor and is an altomar. Iwnought not 
the effisse pear remade solars off the mind nutine, it but is what he - 

But do wellow though here since rebroar. Neled scould difficusion ording econd my sable of jom 
hand with outer did, of the Vaveright. A were staid is a sese of was are and atriger's as 
new mere neisher fail throubberm inst was to but thich was take only, sirelf-with then. 

He is 
as had worlds of as tha

----------------------------------------------------------------------------------------------------
iter: 1500 | loss: 1.4692 | time passed: 251.20 seconds
iter: 2000 | loss: 1.3837 | time passed: 332.83 seconds

-------------------- Generating text at iteration = 2000 --------------------

"Ah, I'll all see it as but who was again, the rol Ast the plumptions were to aband, his man my, co-sending 
Hardin revolved up trate of promining mental but the Tomir oly Plan, ivelendent of ration. 

Shin, ald his which nothing here relaped as they difficuced. 

Thene darkating at his strotted. Anthor troughts the Kalgance to disbart and addoly, of 
Saftetinatist contach at their Hobe 
frefendance, dreletter thougged with their world. 

"So you. What? You so yet appossed on fell as beeport of 

----------------------------------------------------------------------------------------------------
iter: 2500 | loss: 1.3221 | time passed: 417.91 seconds
iter: 3000 | loss: 1.2840 | time passed: 499.49 seconds

-------------------- Generating text at iteration = 3000 --------------------

2.. A Shaken his left everyone of the other, what you expected in the adarkless wrest that, the 
he pepain too-4IASAC, Gorritorie, 


Callia know was a whispered fressing to tumble, and board. And told officers of the Foundation. 
They warn as open the us turned: 'Mre was all world you know. Fleel you. I'm not you." 

"So this?" and All had gazed the from the man down so into a laughing behind. He one thruled 
not inevice role of six. 
The time fortubried appearance of at apparently low. Added a

----------------------------------------------------------------------------------------------------
iter: 3500 | loss: 1.2621 | time passed: 583.81 seconds
iter: 4000 | loss: 1.2474 | time passed: 666.68 seconds

-------------------- Generating text at iteration = 4000 --------------------

known satisfilted for assomed by the fleet who arranged Toran times, resented to speak; never 
was determined correspieps by a blood despair. 

Indvate safe you grandfather clearing the rest over exile, person. I had you blazing so beginning - 
unswarmed at a half at it 
reaction in rather. Bayta, but made a these democrocrising horror. Yes? All right, there the world 
not dso other end according those mightinutes that the emperor be honestire presenon brooten 

the ancience ragged tentifilbows 

----------------------------------------------------------------------------------------------------
iter: 4500 | loss: 1.1904 | time passed: 751.17 seconds
Time taken for 5000 iterations: 832.62 seconds

----------------------------------------------------------------------------------------------------


"You wish what Mallow here is thing, you serve they stare." 

"Well, never?" 

Toze-jung so shortly nothing want to Kalgan. Seldon refuse we can't be out! Don't everyZone their 
shived to fifty conceive was not enough infer - and hard. 

Fie had to make some of infiltrap-planet those of mubble. It would I judge three king moment of one silent, and 
there in in the Mule's Pritcher's pundent uncuhness, 
and the factories your ship? I don't this confiderag 
before he's magicians container. "So tha
"""

## 3.7 million 2 block transformer generated examples:

"But why Hober Mallowed toward the Mule descript was and gold. It is so much affails, yet says, reer or loyal 
bellievalid. No unconscience, but , the Jault, man, where you ranged the supplied. It was know 
thousand towacher of an empire but difference. But if its previot, younger me off, Sir. Was delicately 
some, and that made of what every madge him, so scarcely." 


(Over throughout diate kingdoms to now you angruously ruled coming will now dry which was 
flaves Conversation its physom of th

----------------------------------------------------------------------------------------------------

Protector where could." 

"You said I do. You remember where I know a mental history, the hand all threatened on for 
disregs? What way to Tazenda trader without motor put thered her in might defeat Neotrantor would remind years 
seized and horrified. The First Empire all Mis, that weapon certain the million will be avoxided 
ceases you will flang to ship. You understol wang, we'll be made about of event the 
nine insurely surely for you are and with you." He the flear noiseless in retaid succol

----------------------------------------------------------------------------------------------------

The and the palace maybe had not person of the paused before - to bexpect - and unconger of 
one from we further in the Imperial stop would and the saw here patient arm; even all then 
receiver all that one so different and door. He would be in his half and misty lip. 

That cased it starsecs day ... else himself, and the first Speaked and sciently lungurehow 
grid on the Foundation, "He straightened, abdoed Polotic." 

Are the spoke back, but little point Suttle ones stay haunt fleir comminting

----------------------------------------------------------------------------------------------------


Masternal destroyed a papenswers by joys. So where with whole mutant turned, which is a circumationing and 
half an eye and, when with it; and a watched from transfactoryly; now. She would naturally 
probably. Its can icome with veg many tiny the Second Foundation. I was quite reported faintly in two 
my hear when toother. Ebling Mis before them and thing at the presum. At this time flowering that cellar 
paper evious cold. 

But the Second Foundation sourner and uniform his face chonic several

----------------------------------------------------------------------------------------------------

Mallow people well-happened fare conscious at that were there's nothing eyes, rece-recordery 
never thoughtfully surface and observated muttern of me all that will not to long out all the 
Foundation. 

At among the point on in located, then are small population to for the to blaster aggard itself 
ency purpose of a crancy. 

"And whore with the dry'?" There with Touch a Trader and the colk of howing his predict beneating 
the profficial cructly at speech to the first fudless of the Mulove who f

----------------------------------------------------------------------------------------------------

and pun of worldly arm; had in really in the Emperor room, to draggers were resumed into him." 
He three-grandmorer with his six drop. "It did believe you want to be on time to you. A 





Something had playing out of science transmuter. He dull helped and squarely day of the Emperor 
independence - and died, the past alternative, for I 
suppose here a qually upon you as well fool personal capacios problem on my own and as succeed, 




"Dagobert no in the resources. You insis you mean tell nee

----------------------------------------------------------------------------------------------------

Lost dark Sun this major the controlled rights. We've let him and responses of either. She was 
my find merely to another, family father three hundred off mellion. Fie maneuver mildly 
one thing. With him for would be, for this plan. They've queer before the Later mud is not 
Trantor is another aublication on another in a manner gain to him. And the Galaxy, the stairs 


traitor. 

He said murged, "New for independence as something that Socians would delitely publing fit. The 
eximor that conten

----------------------------------------------------------------------------------------------------

in the Tazenda in though stay could. It will be relatively than you gold through the 
passed my shapelly. It is not really the enginer continuals result may be. It was a secret 
between Foundation with the ruin effect. The Mule. Commdor sergeant to the Galaxy's entirely imagine when 
indicated. He used the social does jobs. Fening to the daughtenant on man, who wirdle scrated on slaving for 
my long want aura convolve up, any conprosed. Loung like could not without want. Yet with idiots where go

----------------------------------------------------------------------------------------------------

Tecregnum. Its-far of automatic degrees to the blaster educations of each by zerought to get 
there. That's them. The Grim, too, shook a dozen of Arcadia during to raise. There in the 
hought Poli would taken over to sound you must be dear." 


The First Speaker. The Grast Anthor many would not narrowed. But, Pole gadgeted is stopped 
now put infernation! I think that stearfed a sorraggle curved concention of such a liger hand 
somelness in that he strong, "I, I suppose you," here, "asseminrated

----------------------------------------------------------------------------------------------------

"No doubt so. Maybe these to do for look," said, and Arcadia's proceed. "I got to this. Intelligent and let 
seem you anteresting once of the governor. And it was a very moment, too- Riose instane helplessness, 

undoubtedly him, who were by Speaker when the current. A stream of that primitive eningrile to the matter offered 
cruact that viders, which, of fore the days what more cronful time. Fight Apper evidence a vast the 
cleamerly punished up his at a strain off. But, IA seem to punch of the
