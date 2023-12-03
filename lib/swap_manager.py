from numpy import dtype
from numpy.core.numeric import full
import torch
import time
import torch.multiprocessing as python_multiprocessing
from multiprocessing import shared_memory
import asyncio
from torchvision import transforms
import os
import random
import pickle
import gc

from PIL import Image
import itertools
import numpy as np

import threading
import queue
from array import array 
import io
import math

from lib.save import DataSaver
from collections import Counter
import directio, io, os
US8K_INDEX=[324, 950, 950, 379, 950, 950, 950, 950, 879, 950]
DSADS_INDEX=[384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384]
IMG1000_INDEX={0: 1300, 1: 1300, 2: 1300, 3: 1300, 4: 1300, 5: 1300, 6: 1300, 7: 1300, 8: 1300, 9: 1300, 10: 1300, 11: 1300, 12: 1300, 13: 1300, 14: 1300, 15: 1300, 16: 1300, 17: 1300, 18: 1300, 19: 1300, 20: 1300, 21: 1300, 22: 1300, 23: 1300, 24: 1300, 25: 1300, 26: 1300, 27: 1300, 28: 1300, 29: 1070, 30: 1300, 31: 1034, 32: 1165, 33: 1300, 34: 1300, 35: 1300, 36: 1300, 37: 1300, 38: 1300, 39: 1300, 40: 1300, 41: 1300, 42: 1300, 43: 1300, 44: 1300, 45: 1300, 46: 1300, 47: 1300, 48: 1300, 49: 1300, 50: 1300, 51: 1300, 52: 1300, 53: 1300, 54: 1300, 55: 1300, 56: 1300, 57: 1300, 58: 1300, 59: 1282, 60: 1300, 61: 1300, 62: 1300, 63: 1300, 64: 1300, 65: 1300, 66: 1300, 67: 1300, 68: 1300, 69: 1300, 70: 1300, 71: 1300, 72: 1300, 73: 1071, 74: 1300, 75: 1300, 76: 1300, 77: 1267, 78: 1300, 79: 1300, 80: 1300, 81: 1300, 82: 1300, 83: 1300, 84: 1300, 85: 1300, 86: 1300, 87: 1300, 88: 1300, 89: 1216, 90: 1300, 91: 1213, 92: 1300, 93: 1300, 94: 1300, 95: 1238, 96: 1300, 97: 1300, 98: 1300, 99: 1300, 100: 1300, 101: 1300, 102: 1300, 103: 1300, 104: 1300, 105: 1300, 106: 1300, 107: 1300, 108: 1300, 109: 1300, 110: 1300, 111: 1176, 112: 1300, 113: 980, 114: 1300, 115: 1300, 116: 1300, 117: 1300, 118: 1300, 119: 1300, 120: 1300, 121: 1300, 122: 1300, 123: 1300, 124: 1300, 125: 1300, 126: 1300, 127: 1300, 128: 1300, 129: 1300, 130: 1300, 131: 1300, 132: 1300, 133: 1300, 134: 1300, 135: 1300, 136: 1300, 137: 1300, 138: 1300, 139: 1300, 140: 1207, 141: 891, 142: 1300, 143: 1300, 144: 1254, 145: 1300, 146: 1300, 147: 1045, 148: 1300, 149: 1300, 150: 1300, 151: 1300, 152: 1300, 153: 1300, 154: 1300, 155: 1300, 156: 1300, 157: 1300, 158: 1300, 159: 1300, 160: 1266, 161: 1300, 162: 1300, 163: 1300, 164: 1300, 165: 1300, 166: 1300, 167: 1300, 168: 1300, 169: 1300, 170: 1300, 171: 1300, 172: 1300, 173: 1253, 174: 1154, 175: 1300, 176: 1300, 177: 1300, 178: 1300, 179: 1300, 180: 1300, 181: 1300, 182: 1300, 183: 1300, 184: 1300, 185: 1300, 186: 969, 187: 1300, 188: 1300, 189: 1300, 190: 1300, 191: 1300, 192: 1272, 193: 1300, 194: 1300, 195: 1300, 196: 1300, 197: 1300, 198: 1292, 199: 1300, 200: 1300, 201: 1300, 202: 1300, 203: 1300, 204: 1300, 205: 1300, 206: 1300, 207: 908, 208: 1300, 209: 1154, 210: 1300, 211: 1300, 212: 1155, 213: 1137, 214: 1300, 215: 1300, 216: 1300, 217: 1300, 218: 1300, 219: 1300, 220: 1300, 221: 1300, 222: 1300, 223: 1300, 224: 1300, 225: 1300, 226: 1300, 227: 1300, 228: 1300, 229: 1300, 230: 1300, 231: 1206, 232: 1300, 233: 1300, 234: 1300, 235: 1300, 236: 1300, 237: 1300, 238: 1300, 239: 1300, 240: 1300, 241: 1300, 242: 1300, 243: 1300, 244: 1199, 245: 1300, 246: 1300, 247: 1300, 248: 1300, 249: 1300, 250: 1300, 251: 1300, 252: 1300, 253: 1300, 254: 1300, 255: 1300, 256: 1300, 257: 1300, 258: 1300, 259: 1300, 260: 1300, 261: 1300, 262: 1300, 263: 1300, 264: 1300, 265: 1300, 266: 1300, 267: 1300, 268: 1300, 269: 1300, 270: 1300, 271: 1300, 272: 1300, 273: 1004, 274: 1300, 275: 1300, 276: 1300, 277: 1300, 278: 1300, 279: 1300, 280: 1300, 281: 1300, 282: 1300, 283: 1300, 284: 1300, 285: 1299, 286: 1300, 287: 1300, 288: 1153, 289: 1290, 290: 1300, 291: 1300, 292: 1300, 293: 1300, 294: 1300, 295: 1300, 296: 1300, 297: 1300, 298: 1300, 299: 1300, 300: 1300, 301: 1300, 302: 1300, 303: 1180, 304: 1300, 305: 1300, 306: 1300, 307: 977, 308: 1300, 309: 738, 310: 1300, 311: 1300, 312: 1300, 313: 1300, 314: 1300, 315: 1300, 316: 1300, 317: 1133, 318: 1300, 319: 1300, 320: 1300, 321: 1272, 322: 1300, 323: 1300, 324: 1300, 325: 1300, 326: 1300, 327: 1300, 328: 1300, 329: 1300, 330: 1300, 331: 1300, 332: 1300, 333: 1300, 334: 1300, 335: 1300, 336: 1300, 337: 1300, 338: 1300, 339: 1300, 340: 1300, 341: 1300, 342: 1300, 343: 1300, 344: 1159, 345: 1300, 346: 1300, 347: 1300, 348: 1300, 349: 1300, 350: 1300, 351: 1300, 352: 1300, 353: 1300, 354: 1300, 355: 1300, 356: 1300, 357: 1300, 358: 1300, 359: 1300, 360: 1300, 361: 1300, 362: 1300, 363: 1300, 364: 1300, 365: 1300, 366: 1300, 367: 1300, 368: 1300, 369: 1300, 370: 1300, 371: 1300, 372: 1300, 373: 1300, 374: 1300, 375: 1300, 376: 1300, 377: 1300, 378: 1300, 379: 1300, 380: 1300, 381: 1300, 382: 1300, 383: 1300, 384: 1300, 385: 1273, 386: 1300, 387: 1300, 388: 1300, 389: 1300, 390: 1300, 391: 1300, 392: 1300, 393: 1300, 394: 1300, 395: 1300, 396: 1300, 397: 1300, 398: 969, 399: 1300, 400: 1300, 401: 1025, 402: 1300, 403: 1250, 404: 1300, 405: 1300, 406: 1300, 407: 1300, 408: 1300, 409: 1300, 410: 1300, 411: 889, 412: 1300, 413: 1300, 414: 1300, 415: 1300, 416: 1300, 417: 1300, 418: 1053, 419: 1300, 420: 1300, 421: 1300, 422: 1300, 423: 732, 424: 1300, 425: 1300, 426: 1300, 427: 1300, 428: 1300, 429: 1300, 430: 1300, 431: 1300, 432: 1300, 433: 1300, 434: 1300, 435: 1300, 436: 1300, 437: 1300, 438: 1300, 439: 1300, 440: 1300, 441: 1300, 442: 1152, 443: 1300, 444: 1300, 445: 1300, 446: 1141, 447: 1300, 448: 1300, 449: 1300, 450: 1300, 451: 1300, 452: 1300, 453: 1300, 454: 1300, 455: 1300, 456: 1300, 457: 1300, 458: 1300, 459: 1300, 460: 1300, 461: 1300, 462: 1084, 463: 1300, 464: 1300, 465: 1300, 466: 1300, 467: 1300, 468: 1300, 469: 1300, 470: 1300, 471: 1300, 472: 1300, 473: 1300, 474: 1300, 475: 1300, 476: 1300, 477: 1300, 478: 1300, 479: 1300, 480: 1300, 481: 1300, 482: 1300, 483: 1150, 484: 1300, 485: 1300, 486: 1300, 487: 1300, 488: 1300, 489: 1300, 490: 1300, 491: 1300, 492: 1300, 493: 1300, 494: 1300, 495: 1300, 496: 1300, 497: 1300, 498: 1300, 499: 1300, 500: 1300, 501: 1300, 502: 1300, 503: 1300, 504: 1300, 505: 1300, 506: 1300, 507: 1300, 508: 1300, 509: 1300, 510: 1300, 511: 1300, 512: 1300, 513: 1300, 514: 1300, 515: 1300, 516: 1239, 517: 1300, 518: 1300, 519: 1300, 520: 1300, 521: 1300, 522: 1258, 523: 1300, 524: 1300, 525: 1300, 526: 1300, 527: 1300, 528: 1300, 529: 1300, 530: 1206, 531: 1300, 532: 1300, 533: 1187, 534: 1300, 535: 1300, 536: 1300, 537: 1300, 538: 1300, 539: 1300, 540: 1300, 541: 1300, 542: 1300, 543: 1300, 544: 1300, 545: 1300, 546: 1300, 547: 1300, 548: 1300, 549: 1300, 550: 1300, 551: 1300, 552: 1300, 553: 1194, 554: 1300, 555: 1300, 556: 1300, 557: 1300, 558: 1300, 559: 1300, 560: 1300, 561: 1300, 562: 1300, 563: 1300, 564: 1300, 565: 1300, 566: 1300, 567: 1300, 568: 1300, 569: 1300, 570: 1300, 571: 1300, 572: 1300, 573: 1300, 574: 1300, 575: 1300, 576: 1300, 577: 1300, 578: 954, 579: 1300, 580: 1300, 581: 1300, 582: 1300, 583: 1300, 584: 1300, 585: 1300, 586: 1300, 587: 1300, 588: 1300, 589: 1300, 590: 1300, 591: 1300, 592: 1300, 593: 1300, 594: 1300, 595: 1300, 596: 1300, 597: 1300, 598: 1300, 599: 1300, 600: 1193, 601: 1300, 602: 1300, 603: 1300, 604: 1300, 605: 1118, 606: 1300, 607: 1300, 608: 1300, 609: 1300, 610: 1300, 611: 1300, 612: 1300, 613: 755, 614: 1300, 615: 1300, 616: 1149, 617: 1300, 618: 1300, 619: 1300, 620: 1300, 621: 1300, 622: 1300, 623: 1300, 624: 1300, 625: 1300, 626: 1300, 627: 1160, 628: 1300, 629: 1300, 630: 1300, 631: 1300, 632: 1186, 633: 1247, 634: 1300, 635: 1300, 636: 1300, 637: 1300, 638: 1300, 639: 1300, 640: 1300, 641: 1300, 642: 1300, 643: 1300, 644: 1222, 645: 1300, 646: 1300, 647: 1300, 648: 1300, 649: 1300, 650: 1300, 651: 1300, 652: 1300, 653: 1300, 654: 1300, 655: 1300, 656: 1300, 657: 1300, 658: 1300, 659: 1300, 660: 1300, 661: 1217, 662: 1300, 663: 1300, 664: 1300, 665: 936, 666: 1300, 667: 1300, 668: 1300, 669: 1300, 670: 1300, 671: 1300, 672: 1300, 673: 1300, 674: 1300, 675: 1300, 676: 1149, 677: 1300, 678: 1029, 679: 1300, 680: 1300, 681: 1120, 682: 1300, 683: 1300, 684: 1300, 685: 1125, 686: 1300, 687: 1300, 688: 1300, 689: 1300, 690: 1300, 691: 1285, 692: 1300, 693: 1300, 694: 1300, 695: 1300, 696: 1300, 697: 1300, 698: 1300, 699: 1300, 700: 1300, 701: 1300, 702: 1300, 703: 1236, 704: 1300, 705: 1300, 706: 1300, 707: 1300, 708: 1300, 709: 1300, 710: 1300, 711: 1300, 712: 1300, 713: 1300, 714: 1300, 715: 1300, 716: 1300, 717: 1300, 718: 1300, 719: 1300, 720: 1300, 721: 1156, 722: 1300, 723: 1300, 724: 1259, 725: 1300, 726: 1300, 727: 1300, 728: 1300, 729: 1300, 730: 1300, 731: 1300, 732: 1300, 733: 1300, 734: 1300, 735: 1300, 736: 962, 737: 1300, 738: 1117, 739: 1300, 740: 1300, 741: 1300, 742: 1300, 743: 1300, 744: 1053, 745: 1300, 746: 1300, 747: 1300, 748: 1300, 749: 1300, 750: 1300, 751: 1300, 752: 1300, 753: 1300, 754: 1300, 755: 1137, 756: 1300, 757: 1300, 758: 1300, 759: 1300, 760: 1300, 761: 1300, 762: 931, 763: 1300, 764: 1300, 765: 1300, 766: 1300, 767: 1300, 768: 1300, 769: 1300, 770: 1300, 771: 1300, 772: 1300, 773: 1300, 774: 1300, 775: 1300, 776: 1067, 777: 1300, 778: 1300, 779: 1300, 780: 1300, 781: 1300, 782: 1300, 783: 1300, 784: 1300, 785: 1300, 786: 1300, 787: 1300, 788: 1300, 789: 1300, 790: 1300, 791: 1300, 792: 1300, 793: 1300, 794: 1300, 795: 1300, 796: 1300, 797: 1300, 798: 1300, 799: 1300, 800: 1300, 801: 1300, 802: 1300, 803: 1270, 804: 1069, 805: 1300, 806: 1300, 807: 1300, 808: 1300, 809: 1300, 810: 1300, 811: 1300, 812: 1300, 813: 1300, 814: 1300, 815: 1300, 816: 1300, 817: 1300, 818: 1300, 819: 1300, 820: 1300, 821: 1300, 822: 1300, 823: 1300, 824: 1300, 825: 1300, 826: 1300, 827: 1300, 828: 1300, 829: 1300, 830: 1300, 831: 1300, 832: 1029, 833: 1300, 834: 1300, 835: 1300, 836: 1300, 837: 1300, 838: 1300, 839: 1300, 840: 1300, 841: 986, 842: 1209, 843: 1300, 844: 1300, 845: 1300, 846: 1300, 847: 1300, 848: 1300, 849: 1300, 850: 1300, 851: 1300, 852: 1162, 853: 1097, 854: 1300, 855: 1300, 856: 1300, 857: 1300, 858: 1300, 859: 1300, 860: 1300, 861: 1300, 862: 1282, 863: 1300, 864: 1300, 865: 1300, 866: 976, 867: 1300, 868: 1300, 869: 1283, 870: 754, 871: 1300, 872: 1300, 873: 1300, 874: 1300, 875: 1300, 876: 1300, 877: 1300, 878: 1300, 879: 1300, 880: 1300, 881: 1300, 882: 1300, 883: 1249, 884: 1300, 885: 1300, 886: 1300, 887: 1300, 888: 1300, 889: 1300, 890: 1300, 891: 1300, 892: 1300, 893: 1300, 894: 1211, 895: 1300, 896: 1300, 897: 1300, 898: 1300, 899: 1300, 900: 1300, 901: 1300, 902: 1300, 903: 1300, 904: 1300, 905: 1300, 906: 1062, 907: 1300, 908: 1300, 909: 1300, 910: 1300, 911: 1300, 912: 1300, 913: 1300, 914: 1136, 915: 1300, 916: 1300, 917: 1300, 918: 1300, 919: 1300, 920: 1300, 921: 1300, 922: 1300, 923: 1300, 924: 1300, 925: 1300, 926: 1300, 927: 1300, 928: 1300, 929: 1157, 930: 1300, 931: 1153, 932: 1300, 933: 1300, 934: 1300, 935: 1218, 936: 1300, 937: 1300, 938: 1300, 939: 1005, 940: 1156, 941: 772, 942: 1300, 943: 1300, 944: 1264, 945: 1300, 946: 1300, 947: 1300, 948: 1300, 949: 1300, 950: 1300, 951: 1300, 952: 1300, 953: 1300, 954: 1300, 955: 1300, 956: 1300, 957: 1300, 958: 1300, 959: 1300, 960: 1300, 961: 1300, 962: 1300, 963: 1300, 964: 1300, 965: 1300, 966: 860, 967: 1117, 968: 1155, 969: 1300, 970: 1300, 971: 1300, 972: 1300, 973: 1136, 974: 1300, 975: 1300, 976: 1300, 977: 1300, 978: 1300, 979: 1300, 980: 1300, 981: 1055, 982: 1300, 983: 1300, 984: 1300, 985: 1300, 986: 1300, 987: 1300, 988: 1300, 989: 1300, 990: 1300, 991: 1300, 992: 1300, 993: 1300, 994: 1300, 995: 1300, 996: 1300, 997: 1300, 998: 1300, 999: 1300}
from torchvision.transforms import RandomResizedCrop
class _DatasetSwapper_CIFAR(object):
    def __init__(self, dataset, saver,testset='cifar100',task_id=0):
        print("[Init Swapper]")
        self.task_id = task_id
        if type(dataset) is not dict and hasattr(dataset, 'replay_dataset'):
            self.dataset = dataset.replay_dataset
        else:
            self.dataset = dataset
        self.saver = saver
        self.t = transforms.ToTensor()
        self.transform = dataset['transform'] 

        self.data_shm_name = dataset['data_shm_name']
        self.targets_shm_name = dataset['targets_shm_name']
        self.filenames_shm_name = dataset['filenames_shm_name']
        
        self.data_shm_list = shared_memory.ShareableList(name=self.data_shm_name) 
        self.targets = shared_memory.ShareableList(name=self.targets_shm_name) 
        self.filenames = shared_memory.ShareableList(name=self.filenames_shm_name) 
        self.testset=testset
        if testset in ["twentynews", "dailynsports", "shakespeare","audioset"]: self.suffix = '.npy'
        else: self.suffix = '.png'
    def update_meta(self, meta):
        self.dataset=meta
        self.data_shm_name = self.dataset['data_shm_name']
        self.targets_shm_name = self.dataset['targets_shm_name']
        self.filenames_shm_name = self.dataset['filenames_shm_name']
        
        self.data_shm_list = shared_memory.ShareableList(name=self.data_shm_name) 
        self.targets = shared_memory.ShareableList(name=self.targets_shm_name) 
        self.filenames = shared_memory.ShareableList(name=self.filenames_shm_name) 
    async def _get_logit(self, logit_filename):
        f = os.open( logit_filename, os.O_RDONLY | os.O_DIRECT)
        os.lseek(f,0,0)
        actual_size = os.path.getsize(logit_filename)
        block_size = 512 * math.ceil(actual_size / 512)
        fr = directio.read(f, block_size)
        os.close(f)
        
        data = io.BytesIO(fr[:actual_size])
        
        logit = pickle.load(data)

        return logit

    async def _get_img(self, filename):
        f = os.open( filename, os.O_RDONLY | os.O_DIRECT)

        os.lseek(f,0,0)
        actual_size = os.path.getsize(filename)
        block_size = 512 * math.ceil(actual_size / 512)
        try:
            fr = directio.read(f, block_size)
        except Exception:
            print(f'Failed reading {filename}',flush=True) 
        os.close(f)
        
        data = io.BytesIO(fr[:actual_size])
        

        img = Image.open(data)
        img = img.convert('RGB')
        
        return img
    
    
    async def _get_file_list(self, path):
        list_dir = os.listdir(path)
        return list_dir

    async def _get_data(self, idx, filename, data_id=None):
        if 'logits' in self.dataset:
            if 'png' in filename:
                logit_filename = filename.split('.png')[0] + '.pkl'
            elif 'pkl' in filename:
                logit_filename = filename
                filename = filename.split('.pkl')[0] + '.png'

            try:
                logit = await self._get_logit(logit_filename)
            except Exception as e:
                print(f'swap_manager._get_data logit: {e}')
                return False
        
        if 'png' in filename or 'JPEG' in filename:
            vec = await self._get_img(filename)
        else:
            vec = np.load(filename)
        swap_in_file = filename.split('/')[-1]
        swap_in_file = swap_in_file.split(self.suffix)[0]

        data_at_idx = shared_memory.SharedMemory(name=self.data_shm_list[idx])
        vec = np.asarray(vec)
        data = np.ndarray(self.dataset['vec_shape'],self.dataset['vec_dtype'],buffer=data_at_idx.buf)
        data[:] = vec
        self.filenames[idx] = swap_in_file 
        
        return True


    async def _swap_main(self, label, swap_idx, swap_in_file=None, data_id=None):
        if swap_in_file is None: #str
            path_curr_label = self.dataset['rb_path'] + '/' + str(label)
        else:
            path_curr_label = self.dataset['rb_path'] + '/' + str(label)
        try:
            if swap_in_file is None:
                prefix = f'{label}_'
                if self.saver is None : 
                    if self.testset=='cifar100':num_file = 500 
                    elif self.testset == 'urbansound8k':
                        num_file = US8K_INDEX[label]
                    elif self.testset=='imagenet1000':
                        num_file = IMG1000_INDEX[label]
                    elif self.testset=='dailynsports':
                        num_file = 384
                    else: num_file=500
                else: num_file = self.saver.get_num_file_for_label_for_swap(label)
                replace_file = path_curr_label + '/' + prefix+str(random.randint(1,num_file)) + self.suffix
            else: 
                replace_file = path_curr_label + '/' + swap_in_file + self.suffix
        except Exception as e:
            print(e)
            return False
        return await self._get_data(swap_idx, replace_file, data_id)
    async def _swap(self, what_to_swap, labels, swap_in_file=None, data_ids=None):
        if data_ids is not None:
            cos = [ self._swap_main(label, idx, swap_in_file, data_id) for label, idx, data_id, swap_in_file in zip(labels, what_to_swap, swap_in_file, data_ids) ]        
        elif swap_in_file is not None and data_ids is None:
            cos = [ self._swap_main(label, idx, swap_in_file) for label, idx, swap_in_file in zip(labels, what_to_swap, swap_in_file) ]

        else: 
            cos = [ self._swap_main(label, idx ) for label, idx in zip(labels, what_to_swap) ]
        res = await asyncio.gather(*cos)
        return res


class _DatasetSwapper_TINY(object):
    def __init__(self, dataset, saver,testset='tiny_imagenet',task_id=0):
        print("[Init Swapper]")
        self.task_id = task_id
        if type(dataset) is not dict and hasattr(dataset, 'replay_dataset'):
            self.dataset = dataset.replay_dataset
        else:
            self.dataset = dataset
        self.saver = saver
        self.t = transforms.ToTensor()
        self.transform = dataset['transform'] 

        self.data_shm_name = dataset['data_shm_name']
        self.targets_shm_name = dataset['targets_shm_name']
        self.filenames_shm_name = dataset['filenames_shm_name']
        
        self.data_shm_list = shared_memory.ShareableList(name=self.data_shm_name) 
        self.targets = shared_memory.ShareableList(name=self.targets_shm_name) 
        self.filenames = shared_memory.ShareableList(name=self.filenames_shm_name) 
        self.testset=testset
        self.suffix = '.png'
    def update_meta(self, meta):
        self.dataset=meta
        self.data_shm_name = self.dataset['data_shm_name']
        self.targets_shm_name = self.dataset['targets_shm_name']
        self.filenames_shm_name = self.dataset['filenames_shm_name']
        
        self.data_shm_list = shared_memory.ShareableList(name=self.data_shm_name) 
        self.targets = shared_memory.ShareableList(name=self.targets_shm_name) 
        self.filenames = shared_memory.ShareableList(name=self.filenames_shm_name) 
    async def _get_logit(self, logit_filename):
        f = os.open( logit_filename, os.O_RDONLY | os.O_DIRECT)
        os.lseek(f,0,0)
        actual_size = os.path.getsize(logit_filename)
        block_size = 512 * math.ceil(actual_size / 512)
        fr = directio.read(f, block_size)
        os.close(f)
        
        data = io.BytesIO(fr[:actual_size])
        
        logit = pickle.load(data)

        return logit

    async def _get_img(self, filename):
        f = os.open( filename, os.O_RDONLY | os.O_DIRECT)

        os.lseek(f,0,0)
        actual_size = os.path.getsize(filename)
        block_size = 512 * math.ceil(actual_size / 512)
        try:
            fr = directio.read(f, block_size)
        except Exception:
            print(f'Failed reading {filename}',flush=True) 
        os.close(f)
        
        data = io.BytesIO(fr[:actual_size])
        

        img = Image.open(data)
        img = img.convert('RGB')
        
        return img
    
    
    async def _get_file_list(self, path):
        list_dir = os.listdir(path)
        return list_dir

    async def _get_data(self, idx, filename, data_id=None):
        if 'logits' in self.dataset:
            if 'png' in filename:
                logit_filename = filename.split('.png')[0] + '.pkl'
            elif 'pkl' in filename:
                logit_filename = filename
                filename = filename.split('.pkl')[0] + '.png'

            try:
                logit = await self._get_logit(logit_filename)
            except Exception as e:
                print(f'swap_manager._get_data logit: {e}')
                
                return False
        
        if 'png' in filename or 'JPEG' in filename:
            vec = await self._get_img(filename)
        else:
            vec = np.load(filename)
        swap_in_file = filename.split('/')[-1]
        swap_in_file = swap_in_file.split(self.suffix)[0]

        data_at_idx = shared_memory.SharedMemory(name=self.data_shm_list[idx])
        vec = np.asarray(vec)
        data = np.ndarray(self.dataset['vec_shape'],self.dataset['vec_dtype'],buffer=data_at_idx.buf)
        data[:] = vec
        self.filenames[idx] = swap_in_file 
        
        return True


    async def _swap_main(self, label, swap_idx, swap_in_file=None, data_id=None):
        if swap_in_file is None: 
            path_curr_label = self.dataset['rb_path'] + '/' + str(label)
        else:
            path_curr_label = self.dataset['rb_path'] + '/' + str(label)
        try:
            if swap_in_file is None:
                prefix = f'{label}_'
                if self.saver is None : num_file=500
                else: num_file = self.saver.get_num_file_for_label_for_swap(label)
                replace_file = path_curr_label + '/' + prefix+str(random.randint(1,num_file)) + self.suffix
            else: 
                replace_file = path_curr_label + '/' + swap_in_file + self.suffix
        except Exception as e:
            print(e)
            return False
        return await self._get_data(swap_idx, replace_file, data_id)
    async def _swap(self, what_to_swap, labels, swap_in_file=None, data_ids=None):
        if data_ids is not None:
            cos = [ self._swap_main(label, idx, swap_in_file, data_id) for label, idx, data_id, swap_in_file in zip(labels, what_to_swap, swap_in_file, data_ids) ]        
        elif swap_in_file is not None and data_ids is None:
            cos = [ self._swap_main(label, idx, swap_in_file) for label, idx, swap_in_file in zip(labels, what_to_swap, swap_in_file) ]

        else: 
            cos = [ self._swap_main(label, idx ) for label, idx in zip(labels, what_to_swap) ]
        res = await asyncio.gather(*cos)
        return res
class _DatasetSwapper_US8K(object):
    def __init__(self, dataset, saver,testset='urbansound8k',task_id=0):
        print("[Init Swapper]")
        self.task_id = task_id
        if type(dataset) is not dict and hasattr(dataset, 'replay_dataset'):
            self.dataset = dataset.replay_dataset
        else:
            self.dataset = dataset
        self.saver = saver
        self.transform = dataset['transform'] 

        self.data_shm_name = dataset['data_shm_name']
        self.targets_shm_name = dataset['targets_shm_name']
        self.filenames_shm_name = dataset['filenames_shm_name']
        
        self.data_shm_list = shared_memory.ShareableList(name=self.data_shm_name) 
        self.targets = shared_memory.ShareableList(name=self.targets_shm_name) 
        self.filenames = shared_memory.ShareableList(name=self.filenames_shm_name) 
        self.testset=testset
        self.suffix = '.png'
    def update_meta(self, meta):
        self.dataset=meta
        self.data_shm_name = self.dataset['data_shm_name']
        self.targets_shm_name = self.dataset['targets_shm_name']
        self.filenames_shm_name = self.dataset['filenames_shm_name']
        
        self.data_shm_list = shared_memory.ShareableList(name=self.data_shm_name) 
        self.targets = shared_memory.ShareableList(name=self.targets_shm_name) 
        self.filenames = shared_memory.ShareableList(name=self.filenames_shm_name) 

    async def _get_img(self, filename):
        f = os.open( filename, os.O_RDONLY | os.O_DIRECT)

        os.lseek(f,0,0)
        actual_size = os.path.getsize(filename)
        block_size = 512 * math.ceil(actual_size / 512)
        try:
            fr = directio.read(f, block_size)
        except Exception:
            print(f'Failed reading {filename}',flush=True) 
        os.close(f)
        
        data = io.BytesIO(fr[:actual_size])
        

        img = Image.open(data)
        img = img.convert('RGB')
        
        return img
    
    
    async def _get_file_list(self, path):
        list_dir = os.listdir(path)
        return list_dir

    async def _get_data(self, idx, filename, data_id=None):
        if 'png' in filename or 'JPEG' in filename:
            vec = await self._get_img(filename)
        else:
            vec = np.load(filename)
        swap_in_file = filename.split('/')[-1]
        swap_in_file = swap_in_file.split(self.suffix)[0]

        data_at_idx = shared_memory.SharedMemory(name=self.data_shm_list[idx])
        vec = np.asarray(vec)
        data = np.ndarray(self.dataset['vec_shape'],self.dataset['vec_dtype'],buffer=data_at_idx.buf)
        data[:] = vec
        self.filenames[idx] = swap_in_file 
        
        return True

    async def _swap_main(self, label, swap_idx, swap_in_file=None, data_id=None):
        if swap_in_file is None: #str
            path_curr_label = self.dataset['rb_path'] + '/' + str(label)
        else:
            # path_curr_label = self.dataset['rb_path']+ '/' + swap_in_file.split('_')[0]
            path_curr_label = self.dataset['rb_path'] + '/' + str(label)
        try:
            if swap_in_file is None:
                prefix = f'{label}_'
                if self.saver is None : 
                    if self.testset=='cifar100':num_file = 500 
                    elif self.testset == 'urbansound8k':
                        num_file = US8K_INDEX[label]
                    elif self.testset=='imagenet1000':
                        num_file = IMG1000_INDEX[label]
                    elif self.testset=='dailynsports':
                        num_file = 384
                    else: num_file=500
                else: num_file = self.saver.get_num_file_for_label_for_swap(label)
                replace_file = path_curr_label + '/' + prefix+str(random.randint(1,num_file)) + self.suffix
            else: 
                replace_file = path_curr_label + '/' + swap_in_file + self.suffix
        except Exception as e:
            print(e)
            return False
        return await self._get_data(swap_idx, replace_file, data_id)
    async def _swap(self, what_to_swap, labels, swap_in_file=None, data_ids=None):
        if data_ids is not None:
            cos = [ self._swap_main(label, idx, swap_in_file, data_id) for label, idx, data_id, swap_in_file in zip(labels, what_to_swap, swap_in_file, data_ids) ]        
        elif swap_in_file is not None and data_ids is None:
            cos = [ self._swap_main(label, idx, swap_in_file) for label, idx, swap_in_file in zip(labels, what_to_swap, swap_in_file) ]

        else: 
            cos = [ self._swap_main(label, idx ) for label, idx in zip(labels, what_to_swap) ]
        res = await asyncio.gather(*cos)
        return res

class _DatasetSwapper_DSADS(object):
    def __init__(self, dataset, saver,testset='cifar100',task_id=0):
        print("[Init Swapper]")
        self.task_id = task_id
        if type(dataset) is not dict and hasattr(dataset, 'replay_dataset'):
            self.dataset = dataset.replay_dataset
        else:
            self.dataset = dataset
        self.saver = saver
        self.t = transforms.ToTensor()
        self.transform = dataset['transform'] 

        self.data_shm_name = dataset['data_shm_name']
        self.targets_shm_name = dataset['targets_shm_name']
        self.filenames_shm_name = dataset['filenames_shm_name']
        
        self.data_shm_list = shared_memory.ShareableList(name=self.data_shm_name) 
        self.targets = shared_memory.ShareableList(name=self.targets_shm_name) 
        self.filenames = shared_memory.ShareableList(name=self.filenames_shm_name) 
        self.testset=testset
        if testset in ["twentynews", "dailynsports", "shakespeare","audioset"]: self.suffix = '.npy'
        else: self.suffix = '.png'
    def update_meta(self, meta):
        self.dataset=meta
        self.data_shm_name = self.dataset['data_shm_name']
        self.targets_shm_name = self.dataset['targets_shm_name']
        self.filenames_shm_name = self.dataset['filenames_shm_name']
        
        self.data_shm_list = shared_memory.ShareableList(name=self.data_shm_name) 
        self.targets = shared_memory.ShareableList(name=self.targets_shm_name) 
        self.filenames = shared_memory.ShareableList(name=self.filenames_shm_name) 
    async def _get_logit(self, logit_filename):
        f = os.open( logit_filename, os.O_RDONLY | os.O_DIRECT)
        os.lseek(f,0,0)
        actual_size = os.path.getsize(logit_filename)
        block_size = 512 * math.ceil(actual_size / 512)
        fr = directio.read(f, block_size)
        os.close(f)
        
        data = io.BytesIO(fr[:actual_size])
        
        logit = pickle.load(data)

        return logit

    async def _get_img(self, filename):
        f = os.open( filename, os.O_RDONLY | os.O_DIRECT)

        os.lseek(f,0,0)
        actual_size = os.path.getsize(filename)
        block_size = 512 * math.ceil(actual_size / 512)
        try:
            fr = directio.read(f, block_size)
        except Exception:
            print(f'Failed reading {filename}',flush=True) 
        os.close(f)
        
        data = io.BytesIO(fr[:actual_size])
        

        img = Image.open(data)
        img = img.convert('RGB')
        
        return img
    
    
    async def _get_file_list(self, path):
        list_dir = os.listdir(path)
        return list_dir

    async def _get_data(self, idx, filename, data_id=None):
        if 'logits' in self.dataset:
            if 'png' in filename:
                logit_filename = filename.split('.png')[0] + '.pkl'
            elif 'pkl' in filename:
                logit_filename = filename
                filename = filename.split('.pkl')[0] + '.png'

            try:
                logit = await self._get_logit(logit_filename)
            except Exception as e:
                print(f'swap_manager._get_data logit: {e}')
                
                return False
        
        if 'png' in filename or 'JPEG' in filename:
            vec = await self._get_img(filename)
        else:
            vec = np.load(filename)
        swap_in_file = filename.split('/')[-1]
        swap_in_file = swap_in_file.split(self.suffix)[0]

        data_at_idx = shared_memory.SharedMemory(name=self.data_shm_list[idx])
        # crop to uniform sizes 
        # Marker 
        if self.testset in ['imagenet1000']:       
            try: 
                vec = self.transform(vec)
            except Exception as e: 
                print(f'get_data error{e}')
            vec = np.asarray(vec)

        else: 
            vec = np.array(self.transform(vec))
        data = np.ndarray(self.dataset['img_shape'],self.dataset['img_dtype'],buffer=data_at_idx.buf)
        data[:] = vec
        self.filenames[idx] = swap_in_file 
        
        return True
    async def _swap_main(self, label, swap_idx, swap_in_file=None, data_id=None):
        if swap_in_file is None: #str
            path_curr_label = self.dataset['rb_path'] + '/' + str(label)
        else:
            path_curr_label = self.dataset['rb_path'] + '/' + str(label)
        try:
            if swap_in_file is None:
                prefix = f'{label}_'
                if self.saver is None : 
                    if self.testset=='cifar100':num_file = 500 
                    elif self.testset == 'urbansound8k':
                        num_file = US8K_INDEX[label]
                    elif self.testset=='imagenet1000':
                        num_file = IMG1000_INDEX[label]
                    elif self.testset=='dailynsports':
                        num_file = 384
                    else: num_file=500
                else: num_file = self.saver.get_num_file_for_label_for_swap(label)
                replace_file = path_curr_label + '/' + prefix+str(random.randint(1,num_file)) + self.suffix
            else: 
                replace_file = path_curr_label + '/' + swap_in_file + self.suffix
        except Exception as e:
            print(e)
            return False
        return await self._get_data(swap_idx, replace_file, data_id)
    async def _swap(self, what_to_swap, labels, swap_in_file=None, data_ids=None):
        if data_ids is not None:
            cos = [ self._swap_main(label, idx, swap_in_file, data_id) for label, idx, data_id, swap_in_file in zip(labels, what_to_swap, swap_in_file, data_ids) ]        
        elif swap_in_file is not None and data_ids is None:
            cos = [ self._swap_main(label, idx, swap_in_file) for label, idx, swap_in_file in zip(labels, what_to_swap, swap_in_file) ]

        else: 
            cos = [ self._swap_main(label, idx ) for label, idx in zip(labels, what_to_swap) ]
        res = await asyncio.gather(*cos)
        return res



class SwapManager(object):
    total_count = 0
    overlap_count = 0

    def __init__(self,  replay_dataset,num_workers, swap_base, g_lambda = 0.99, store_budget=None, get_loss=False, get_entropy=False, seed=None, **kwargs):
        self.swap_base = swap_base
        self.swap_determine = self.swap_policy(swap_base)
        self.softmax = torch.nn.Softmax(dim=1)
        self.num_workers = num_workers
        self.dataset = replay_dataset
        
        self.rp_len = shared_memory.ShareableList(name=self.dataset['rp_len_name'])
        self.test_set = replay_dataset['test_set']
        self.num_swap = 0
        self.cls_num_swap = dict()
        
        self.agent = self.dataset['agent']
        self._swap_loss = None
        self.new_classes = None
        self.seed = seed
        self.g_lambda = g_lambda
        self.testset = 'cifar'

        self.swap_class_dist = {}
        
        print(f'Swapper Parameters: {kwargs}')
        if 'dataset' in kwargs:
            if kwargs['dataset'] in ['imagenet1000', 'imagenet100','miniimagenet']:
                self.testset = 'imagenet'
            elif kwargs['dataset'] in ["twentynews", "dailynsports", "shakespeare"]:
                self.testset = kwargs['dataset'] 

        if 'saver' in kwargs and kwargs['saver'] == False:
            self.saver = None 
            print('[Saver]: Disabled')
        else:
            self.saver = DataSaver(replay_dataset['rb_path'], store_budget, seed, self.test_set)
            print('[Creating Saver]...Done')
            print(f'[Saver Path]: {self.saver.rb_path}')
        if num_workers == 0:
            self._swapper =  self.swapper_dict[self.test_set](self.dataset, self.saver,self.test_set)

        if 'threshold' in kwargs:
            self.threshold = float(kwargs['threshold'][0])
            if len(kwargs['threshold'])>1:
                self.threshold2 = float(kwargs['threshold'][1])
            else: 
                self.threshold2 = float(kwargs['threshold'][0])
        
        if 'swap_options' in kwargs:
            self.swap_options = kwargs['swap_options']
        else: 
            self.swap_options = []
        if 'filename' in kwargs:
            self.filename = kwargs['filename']

        if 'result_save_path' in kwargs:
            self.result_save_path = kwargs['result_save_path']
        
        self._get_loss = get_loss
        self._get_entropy = get_entropy

        self.data_correct_entropy = []
        self.data_wrong_entropy = []

        self.data_correct_loss = []
        self.data_wrong_loss = []
        
        self.stream_loss = []
        self.replay_loss = []

        self.swapper_dict = {
            'cifar100':_DatasetSwapper_CIFAR,
            'urbansound8k':_DatasetSwapper_US8K,
            'dailynsports': _DatasetSwapper_DSADS,
            'tiny_imagenet': _DatasetSwapper_TINY
        }

    def update_meta(self,meta):
        self.dataset = meta
        self.swap_pt([-7], meta,filenames=None,data_ids=None)
    def adaptive_swap_loop(self,worker_id, dataset, saver, swap_queue, all_recved, all_swapped,pause_n_reset,cleared,seed=None,num_issued=None,task_id=0):

        if seed is not None:
            np.random.seed(seed + worker_id)
            torch.manual_seed(seed + worker_id)
            random.seed(seed + worker_id)
            torch.cuda.manual_seed_all(seed + worker_id)
            torch.cuda.manual_seed(seed + worker_id)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        swapper = self.swapper_dict[self.test_set](dataset,saver,testset=self.test_set,task_id=task_id)
        
        
        while True:
            try:
                idxs,targets,filenames,data_ids = swap_queue.get(block=False)
                
            except queue.Empty: 
                cleared.set()
                time.sleep(1)
                continue
            cleared.clear()
            if pause_n_reset.is_set(): 
                continue
            
            if idxs is None and targets is None:
                assert all_recved.is_set()
                all_swapped.set()
                self.filenames = shared_memory.ShareableList(name= self.dataset['filenames_shm_name']) 
                break
            elif idxs[0] == -7 : 
                swapper.update_meta(targets)
                continue            
            elif all_recved.is_set():
                continue
            elif all_swapped.is_set():
                break
                
            swap_idx,labels = self.swap_determine(idxs,targets,filenames,data_ids)
            
            swap_idx, labels = swap_idx.tolist(), labels.tolist()
            
            # Determine swap-in file here, if we need to determine it in advance 
            swap_in_file = None        
            if swap_in_file is not None:
                swap_res = asyncio.run(swapper._swap(swap_idx, labels, swap_in_file))
            else:
                swap_res = asyncio.run(swapper._swap(swap_idx, labels))

            num_issued.value += len(swap_idx)
            del swap_idx, labels, swap_in_file,data_ids    
    def set_swap_ratio(self,coeff):
        self.threshold *= coeff
        self.threshold2 *= coeff
        return [self.threshold,self.threshold2]
    def get_swap_ratio(self):
        return np.min([self.threshold,self.threshold2])
    def before_train(self,dataset=None,task_id=0):  
        self.task_id = task_id
        if dataset is not None: 
            self.dataset = dataset
        if self.agent in ["der","derpp", "tiny","aser"]:
            self.rp_len = None
        else:
            self.rp_len = shared_memory.ShareableList(name=self.dataset['rp_len_name'])

        if self.saver is not None:
            saver_pid = self.saver.before_train()
        else: saver_pid = -1

        if self.num_workers > 0 :
            #swap process
            self._swap_workers=[]
            self._swap_queues=[]
            self._issue_done_event = python_multiprocessing.Event()
            self._reset_n_pause = python_multiprocessing.Event()
            self._cleared = python_multiprocessing.Event()
            self._all_swapped_event = python_multiprocessing.Event()
            self._swap_worker_queue_idx_cycle = itertools.cycle(range(self.num_workers))
            self._num_issued = python_multiprocessing.Value('i',0)
            self._swap_worker_pids=[]
            for i in range(self.num_workers):
                swap_queue = python_multiprocessing.Queue()        
                swap_worker = python_multiprocessing.Process(
                    target = self.adaptive_swap_loop,
                    args=(i, self.dataset, self.saver, swap_queue, self._issue_done_event,self._all_swapped_event,self._reset_n_pause,self._cleared,self.seed,self._num_issued,self.task_id)
                )
                swap_worker.daemon = True
                swap_worker.start()
                self._swap_worker_pids.append(swap_worker.pid)
                
                self._swap_workers.append(swap_worker)
                self._swap_queues.append(swap_queue)
        return  self._swap_worker_pids,saver_pid

    def pause(self):
        self._reset_n_pause.set()
        while True:
            if self._cleared.is_set():break
            else: time.sleep(1)

    def resume(self):
        self._reset_n_pause.clear()
        while True:
            if self._cleared.is_set():break
            else: time.sleep(1)
    def after_train(self, get_loss=False, get_entropy=False,now=False):
        #shudown swap process
        if self.num_workers > 0:
            self._issue_done_event.set()
            if now == True: self._all_swapped_event.set()
            for sq in self._swap_queues:
                sq.put((None, None, None, None))
            
            for s in self._swap_workers:
                s.join() #timeout?

            for sq in self._swap_queues:
                sq.cancel_join_thread()
                sq.close()

            gc.collect()
            print("[Swapper shutdown]")
            if self.saver is not None:
                self.saver.after_train()
        
        all_swapped, num_issued =  self._all_swapped_event.is_set(), self._num_issued.value
        #reset events
        self._issue_done_event.clear()
        self._reset_n_pause.clear()
        self._all_swapped_event.clear()
        self._cleared.set()
        return all_swapped, num_issued
    def get_num_swap(self):
        return self.num_swap
    def get_cls_num_swap(self):
        return self.cls_num_swap

    def reset_num_swap(self):
        self.num_swap = 0
        self.cls_num_swap = self.cls_num_swap.fromkeys(self.cls_num_swap,0)

    def reset_swap_class_dist(self):
        self.swap_class_dist = {}

    def swap(self, what_to_swap, labels):

        None_list = [None] * len(what_to_swap)
        if len(what_to_swap) > 0:
            if hasattr(self, "_swapper"):
                asyncio.run(self._swapper._swap(what_to_swap, labels))
            elif hasattr(self, "_swap_queues"):
                worker_id = next(self._swap_worker_queue_idx_cycle)
                self._swap_queues[worker_id].put(
                    (what_to_swap, labels,None_list,None))
            self.num_swap = self.num_swap + len(what_to_swap)
            counter = Counter(labels)
            for key in counter:
                if key not in self.cls_num_swap: self.cls_num_swap[key] = counter[key]
                else: 
                     self.cls_num_swap[key] += counter[key]
                     
    def swap_pt(self, idxs,targets,filenames=None,data_ids=None):  
        
        if len(idxs) > 0:
            swap_in_file = [None] * len(idxs)
            filenames= [None] * len(idxs)
            if hasattr(self, "_swapper"):
                asyncio.run(self._swapper._swap_pt(what_to_swap, labels, swap_in_file))
            elif hasattr(self, "_swap_queues"):
                worker_id = next(self._swap_worker_queue_idx_cycle)
                self._swap_queues[worker_id].put(
                    (idxs,targets,filenames,data_ids))
    def swap_policy(self, swap_base):
        policies = {
            "entropy" : self.entropy,
            "gradtail" : self.gradtail,
            "softmax" : self._softmax,
            "hybrid_threshold" : self.hybrid,
            "prediction" : self.prediction,
            "random" : self.random,
            "random_fixed": self.random_fixed,
            "pure_random" : self.pure_random,
            "hybrid_ratio" : self.hybrid_ratio,
            "hybrid_balanced" : self.hybrid_balanced,
            "hybrid_balanced_p" : self.hybrid_balanced_p,
            "hybrid_loss" : self.hybrid_loss,
            "entropy_based": self.entropy_based,
            "gradient_based": self.gradient_based,
            "all" : self.all
        }
        return policies[swap_base]

    @property
    def swap_thr(self):
        return self._swap_thr
    
    @swap_thr.setter
    def swap_thr(self, thr):
        self._swap_thr = thr

    
    @property
    def swap_loss(self):
        return self._swap_loss
    
    @swap_loss.setter
    def swap_loss(self, loss):
        self._swap_loss = loss
    
    def to_onehot(self, targets, n_classes):
        onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
        onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
        return onehot

    def get_replay_index(self, idxs, targets, data_ids=None):

        if self.rp_len is None:
            if data_ids is not None:
                return idxs, targets, data_ids
            else:
                return idxs, targets

        else:
            replay_index_of_idxs = (idxs < self.rp_len[0]).squeeze().nonzero(as_tuple=True)[0]
            
            if data_ids is not None:
                return idxs[replay_index_of_idxs], targets[replay_index_of_idxs], data_ids[replay_index_of_idxs]
            else:
                return idxs[replay_index_of_idxs], targets[replay_index_of_idxs]

    def prediction(self, idxs, outputs, targets, data_ids=None):
        #
        # determine what to swap based on mis-prediction
        #
        predicts = torch.max(outputs, dim=1)[1]
        selected_idx = (predicts.cpu() == targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        if data_ids is not None:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx], data_ids[selected_idx])
        else:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx])

    def _softmax(self, idxs, outputs, targets, data_ids=None,mode='default'):
        #
        # determine what to swap based on entropy (threshold = 1.0 : lower is easy and swap, higher is hard and preserve)
        #
        if mode == 'default':
            threshold = self.threshold
        else:
            threshold = self.threshold2
        soft_output = self.softmax(outputs)[:,0]
        median = torch.median(soft_output)

        # mid_idx = int(len(soft_output)*self.threshold)
        assert(self.threshold == 0.5)
        selected_idx = (soft_output < median).squeeze().nonzero(as_tuple=True)[0]

        if data_ids is not None:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx], data_ids[selected_idx])
        else:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx])

    #@profile
    
    def random_fixed(self, idxs, outputs, targets, data_ids=None):
        
        swap_ratio = self.threshold
        
        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len[0]).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)
        
        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))
        
        replay_output = outputs[replay_index_of_idxs]
        replay_idxs = idxs[replay_index_of_idxs]
        replay_targets = targets[replay_index_of_idxs]
        
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs]
            return replay_idxs[:how_much_swap], replay_targets[:how_much_swap], replay_data_ids[:how_much_swap]
        else:
            return replay_idxs[:how_much_swap], replay_targets[:how_much_swap]
            
    def random(self, idxs, targets, filenames=None, data_ids=None,mode='default'):
        if mode == 'default':
            swap_ratio = self.threshold
        else:
            swap_ratio = self.threshold2


        # idxs and targets are lists now 
        idxs, targets = torch.tensor(idxs),torch.tensor(targets)
        if swap_ratio == 0: return torch.tensor([]),torch.tensor([])
        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len[0]).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)
        


        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if (self.threshold==1): return replay_idxs, replay_targets
        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))
        selected_index = np.random.choice(len(replay_idxs), how_much_swap, replace=False)

        assert len(selected_index) == how_much_swap

        return replay_idxs[selected_index], replay_targets[selected_index] #, outputs[selected_index]
    def pure_random(self, idxs, outputs, targets, data_ids=None):

        swap_ratio = self.threshold

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len[0]).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))

        idx_to_pick = np.linspace(0, len(replay_index_of_idxs), num=how_much_swap, dtype = int, endpoint=False)

        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        
        soft_output = self.softmax(replay_output)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()

        if replay_output.nelement() != 0:
            predicts = torch.max(replay_output, dim=1)[1]
            r_predicted = (predicts.cpu() == replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]

            r_idxs = replay_idxs[r_predicted]
            r_entropy = entropy[r_predicted]
            r_targets = replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = replay_data_ids[r_predicted]
    
            r_range = np.where(idx_to_pick < len(r_idxs))
            sorted_r = torch.argsort(r_entropy)[idx_to_pick[r_range]]

            
            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]


            if len(sorted_r) < how_much_swap:
                idx_to_pick = np.delete(idx_to_pick, r_range)
                idx_to_pick = idx_to_pick - len(r_idxs)

                w_predicted = (predicts.cpu() != replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                
                w_idxs = replay_idxs[w_predicted]
                w_entropy = entropy[w_predicted]
                w_targets = replay_targets[w_predicted]

                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]
                
                sorted_w = torch.argsort(w_entropy, descending=True)[idx_to_pick]


                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]


                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)
        

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids


        else:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)
        
        assert len(selected_idxs) == how_much_swap

        if data_ids is not None:
            return selected_idxs, selected_targets, selected_data_ids
        else:
            return selected_idxs, selected_targets

    def entropy_based(self, idxs, outputs, targets, filenames=None,data_ids=None,mode='default'):
        if mode == 'default':
            threshold = self.threshold
        else:
            threshold = self.threshold2
            
        # which samples to swap out
        if "hard" in self.swap_options:
            reverse = True 
        elif "easy" in self.swap_options:
            reverse = False 
        else: reverse = False
        # whether diversity is considered or not 
        if "diverse" in self.swap_options:
            diverse = True 
        else: 
            diverse = False 
        # whether we swap out the same ratio for each class
        if "balanced" in self.swap_options:
            balanced= True 
        else: 
            balanced = False 

        assert(not(diverse==True and balanced==True)) # diverse and balanced are mutually exclusive

        if balanced and diverse==False:  # Class balanced 
            if reverse: 
                return self.hybrid_balanced_p_opposite(idxs, outputs, targets, filenames, data_ids,mode)
            else: 
                return self.hybrid_balanced_p(idxs, outputs, targets, filenames, data_ids,mode)
        
        elif balanced==False: # not class balanced 
            if reverse: 
                return self.hybrid_hard(idxs, outputs, targets, filenames, data_ids,mode)
            else:
                return self.hybrid_easy(idxs, outputs, targets, filenames, data_ids,mode)
            

    def entropy(self, idxs, outputs, targets, data_ids=None,mode='default'):
        #
        # determine what to swap based on entropy (threshold = 1.0 : lower is easy and swap, higher is hard and preserve)
        #
        soft_output = self.softmax(outputs)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        selected_idx = (entropy.cpu() < threshold).squeeze().nonzero(as_tuple=True)[0]

        if data_ids is not None:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx], data_ids[selected_idx])
        else:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx])

    def hybrid(self, idxs, outputs, targets, data_ids=None,mode='default'):
        #
        # determine what to swap based on entropy (threshold : lower is easy and swap, higher is hard and preserve)
        #        
        if mode == 'default':
            threshold = self.threshold
        else:
            threshold = self.threshold2
                
        soft_output = self.softmax(outputs)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        entropy_batch = (entropy.cpu() < self.threshold).squeeze()
        #
        # if wrong predicted sample with low entropy, don't make it swap (make swap FALSE)
        #
        predicts = torch.max(outputs, dim=1)[1]
        prediction_batch = (predicts.cpu() == targets.cpu()).squeeze()

        selected_idx = (torch.logical_and(entropy_batch, prediction_batch)).nonzero(as_tuple=True)[0]
        if data_ids is not None:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx], data_ids[selected_idx])
        else:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx])

    def hybrid_balanced_p(self, idxs, outputs, targets, filenames=None,data_ids=None,mode='default'):
        assert('diversity' not in self.swap_options)
        
        #initialize parameters 
        if mode == 'default':
            swap_ratio = self.threshold
        else:
            swap_ratio = self.threshold2


        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len[0]).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        total_how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))

        #get the number of each class inside the batch
        batch_dist = {}
        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        replay_filenames = [filenames[i] for i in replay_index_of_idxs]

        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()
        
        for cls in replay_targets:
            if cls.item() not in batch_dist:
                batch_dist[cls.item()] = 1
            else:
                batch_dist[cls.item()] += 1

        expected = 0
        for key in batch_dist.keys():
            batch_dist[key] = math.modf( batch_dist[key] * swap_ratio )
            expected += int(batch_dist[key][1])

        shortage = total_how_much_swap - expected

        if shortage > 0:
            get_dec_samples = list(filter(lambda x: x[1][0] != 0, batch_dist.items()))
            selected_dec_samples = random.sample(get_dec_samples, shortage)
            
            for t in selected_dec_samples:
                k, _ = t
                #k = t
                batch_dist[k] = tuple((batch_dist[k][0] ,batch_dist[k][1] + 1))
        
        if replay_output.nelement() == 0:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        #separate batch into class-wise
        for i, (k,v) in enumerate(batch_dist.items()):
            how_much_swap = int(v[1])
            if how_much_swap == 0:
                continue
            
            
            cur_cls_idx = (replay_targets==k).squeeze().nonzero(as_tuple=True)[0]
            
            cur_replay_output = replay_output[cur_cls_idx]
            cur_replay_idxs = replay_idxs[cur_cls_idx]
            cur_replay_targets = replay_targets[cur_cls_idx]            
            cur_replay_filenames = [replay_filenames[i] for i in cur_cls_idx]        
            if data_ids is not None:
                cur_replay_data_ids = replay_data_ids[cur_cls_idx]
            
            soft_output = self.softmax(cur_replay_output)
            entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
            
            predicts = torch.max(cur_replay_output, dim=1)[1]
            r_predicted = (predicts.cpu() == cur_replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]

            r_idxs = cur_replay_idxs[r_predicted]
            r_entropy = entropy[r_predicted]
            r_targets = cur_replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = cur_replay_data_ids[r_predicted]


            sorted_r = torch.argsort(r_entropy)[:how_much_swap]



            #to check this code validate
            selected_r_entropy = r_entropy[sorted_r]

            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]

            if len(sorted_r) < how_much_swap:

                w_predicted = (predicts.cpu() != cur_replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                w_idxs = cur_replay_idxs[w_predicted]
                w_entropy = entropy[w_predicted]
                w_targets = cur_replay_targets[w_predicted]
                    
                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]
                
                w_how_much_swap = how_much_swap-len(sorted_r)
                sorted_w = torch.argsort(w_entropy, descending=True)[:w_how_much_swap]
                
                selected_w_entropy = w_entropy[sorted_w]

                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]

                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)
            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids

            try:
                total_selected_idxs = torch.cat((total_selected_idxs,selected_idxs),dim=-1)
                total_selected_targets = torch.cat((total_selected_targets,selected_targets),dim=-1)
                if data_ids is not None:
                    total_selected_data_ids = torch.cat((total_selected_data_ids,selected_data_ids),dim=-1)

            except Exception:                
                total_selected_idxs = selected_idxs
                total_selected_targets = selected_targets
                if data_ids is not None:
                    total_selected_data_ids = selected_data_ids

        assert len(total_selected_idxs) == total_how_much_swap

        if data_ids is not None:
            return total_selected_idxs, total_selected_targets, total_selected_data_ids
        else:
            return total_selected_idxs, total_selected_targets
    def hybrid_balanced_p_opposite(self, idxs, outputs, targets, filenames=None,data_ids=None,mode='default'):
        assert('diversity' not in self.swap_options)
        
        #initialize parameters 
        if mode == 'default':
            swap_ratio = self.threshold
        else:
            swap_ratio = self.threshold2


        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len[0]).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        total_how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))

        #get the number of each class inside the batch
        batch_dist = {}
        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        replay_filenames = [filenames[i] for i in replay_index_of_idxs]

        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()
        
        for cls in replay_targets:
            if cls.item() not in batch_dist:
                batch_dist[cls.item()] = 1
            else:
                batch_dist[cls.item()] += 1

        expected = 0
        for key in batch_dist.keys():
            batch_dist[key] = math.modf( batch_dist[key] * swap_ratio )
            expected += int(batch_dist[key][1])

        shortage = total_how_much_swap - expected
        if shortage > 0:
            get_dec_samples = list(filter(lambda x: x[1][0] != 0, batch_dist.items()))
            selected_dec_samples = random.sample(get_dec_samples, shortage)
            
            for t in selected_dec_samples:
                k, _ = t
                #k = t
                batch_dist[k] = tuple((batch_dist[k][0] ,batch_dist[k][1] + 1))
        
        if replay_output.nelement() == 0:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        #separate batch into class-wise
        for i, (k,v) in enumerate(batch_dist.items()):

            how_much_swap = int(v[1])
            if how_much_swap == 0:
                continue
            
            
            cur_cls_idx = (replay_targets==k).squeeze().nonzero(as_tuple=True)[0]
            
            cur_replay_output = replay_output[cur_cls_idx]
            cur_replay_idxs = replay_idxs[cur_cls_idx]
            cur_replay_targets = replay_targets[cur_cls_idx]            
            if data_ids is not None:
                cur_replay_data_ids = replay_data_ids[cur_cls_idx]
            
            soft_output = self.softmax(cur_replay_output)
            entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
            
            predicts = torch.max(cur_replay_output, dim=1)[1]
            w_predicted = (predicts.cpu() != cur_replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]

            w_idxs = cur_replay_idxs[w_predicted]
            w_entropy = entropy[w_predicted]
            w_targets = cur_replay_targets[w_predicted]
            if data_ids is not None:
                w_data_ids = cur_replay_data_ids[w_predicted]


            sorted_w = torch.argsort(w_entropy,descending=True)[:how_much_swap]



            #to check this code validate
            selected_w_entropy = w_entropy[sorted_w]


            selected_w_idxs = w_idxs[sorted_w]
            selected_w_targets = w_targets[sorted_w]
            if data_ids is not None:
                selected_w_data_ids = w_data_ids[sorted_w]

            if len(sorted_w) < how_much_swap:

                r_predicted = (predicts.cpu() == cur_replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                r_idxs = cur_replay_idxs[r_predicted]
                r_entropy = entropy[r_predicted]
                r_targets = cur_replay_targets[r_predicted]
                    
                if data_ids is not None:
                    r_data_ids = replay_data_ids[r_predicted]

                r_how_much_swap = how_much_swap-len(sorted_w)
                sorted_r = torch.argsort(r_entropy, descending=False)[:r_how_much_swap]
                
                #to check this code validate
                selected_r_entropy = r_entropy[sorted_r]

                selected_r_idxs = r_idxs[sorted_r]
                selected_r_targets = r_targets[sorted_r]
                    
                if data_ids is not None:
                    selected_r_data_ids = r_data_ids[sorted_r]

                selected_idxs = torch.cat((selected_w_idxs,selected_r_idxs),dim=-1)
                selected_targets = torch.cat((selected_w_targets,selected_r_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_w_data_ids,selected_r_data_ids),dim=-1)
            else:
                selected_idxs = selected_w_idxs
                selected_targets = selected_w_targets
                if data_ids is not None:
                    selected_data_ids = selected_w_data_ids

            try:
                total_selected_idxs = torch.cat((total_selected_idxs,selected_idxs),dim=-1)
                total_selected_targets = torch.cat((total_selected_targets,selected_targets),dim=-1)
                if data_ids is not None:
                    total_selected_data_ids = torch.cat((total_selected_data_ids,selected_data_ids),dim=-1)

            except Exception:                
                total_selected_idxs = selected_idxs
                total_selected_targets = selected_targets
                if data_ids is not None:
                    total_selected_data_ids = selected_data_ids

        assert len(total_selected_idxs) == total_how_much_swap

        if data_ids is not None:
            return total_selected_idxs, total_selected_targets, total_selected_data_ids
        else:
            return total_selected_idxs, total_selected_targets

    def hybrid_balanced(self, idxs, outputs, targets, data_ids=None):
        
        swap_ratio = self.threshold

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len[0]).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))

        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        soft_output = self.softmax(replay_output)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()

        if replay_output.nelement() != 0:
            predicts = torch.max(replay_output, dim=1)[1]
            r_predicted = (predicts.cpu() == replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]

            r_idxs = replay_idxs[r_predicted]
            r_entropy = entropy[r_predicted]
            r_targets = replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = replay_data_ids[r_predicted]

            sorted_r_org = torch.argsort(r_entropy)

            selected = []
            filled_counter = 0

            for i, idx in enumerate(sorted_r_org):

                if filled_counter >= how_much_swap:
                    break

                label = r_targets[idx].item()
                if label in self.swap_class_dist:
                    if self.swap_class_dist[label] + 1 <= self.swap_thr:
                        self.swap_class_dist[label] += 1
                        filled_counter +=1
                        selected.append(i)
                    else:
                        continue
                else:
                    self.swap_class_dist[label] = 1
                    filled_counter +=1
                    selected.append(i)

            sorted_r = sorted_r_org[selected][:how_much_swap]

            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]

            if len(sorted_r) < how_much_swap:

                w_predicted = (predicts.cpu() != replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                w_idxs = replay_idxs[w_predicted]
                w_entropy = entropy[w_predicted]
                w_targets = replay_targets[w_predicted]
                    
                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]

                w_how_much_swap = how_much_swap-len(sorted_r)
                sorted_w_org = torch.argsort(w_entropy, descending=True)

                selected = []
                filled_counter = 0

                for i, idx in enumerate(sorted_w_org):
                    if filled_counter >= w_how_much_swap:
                        break
                    label = w_targets[idx].item()
                    if label in self.swap_class_dist:
                        if self.swap_class_dist[label] + 1 <= self.swap_thr:
                            self.swap_class_dist[label] += 1
                            filled_counter +=1
                            selected.append(i)
                        else:
                            continue
                    else:
                        self.swap_class_dist[label] = 1
                        filled_counter +=1
                        selected.append(i)
                    
                sorted_w = sorted_w_org[selected][:w_how_much_swap]
                    
                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]

                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids
        
        else:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        if len(selected_idxs) != how_much_swap:
            for unselected in sorted_r_org:
                
                if len(selected_idxs) == how_much_swap:
                    break

                if unselected not in sorted_r:
                    selected_idxs = torch.cat((selected_idxs,r_idxs[unselected].reshape(1)),dim=-1)
                    selected_targets = torch.cat((selected_targets,r_targets[unselected].reshape(1)),dim=-1)
                    if data_ids is not None:
                        selected_data_ids = torch.cat((selected_data_ids,r_data_ids[unselected].reshape(1)),dim=-1)
                    self.swap_class_dist[r_targets[unselected].item()] += 1
            
            for unselected in sorted_w_org:
                if len(selected_idxs) == how_much_swap:
                    break

                if unselected not in sorted_w:
                    selected_idxs = torch.cat((selected_idxs,w_idxs[unselected].reshape(1)),dim=-1)
                    selected_targets = torch.cat((selected_targets,w_targets[unselected].reshape(1)),dim=-1)
                    if data_ids is not None:
                        selected_data_ids = torch.cat((selected_data_ids,w_data_ids[unselected].reshape(1)),dim=-1)
                    self.swap_class_dist[w_targets[unselected].item()] += 1


        assert len(selected_idxs) == how_much_swap

        if data_ids is not None:
            return selected_idxs, selected_targets, selected_data_ids
        else:
            return selected_idxs, selected_targets

    def hybrid_hard(self, idxs, outputs, targets,filenames=None,data_ids=None,mode='default'):
        assert('balanced' not in self.swap_options)
        #initialize parameters 
        if mode == 'default':
            swap_ratio = self.threshold
        else:
            swap_ratio = self.threshold2
        # whether diversity is considered or not 
        if "diverse" in self.swap_options:
            diverse = True 
        else: 
            diverse = False 

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len[0]).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))

        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        soft_output = self.softmax(replay_output)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        
        if replay_output.nelement() != 0:
            predicts = torch.max(replay_output, dim=1)[1]
            w_predicted = (predicts.cpu() != replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]

            w_idxs = replay_idxs[w_predicted]
            w_entropy = entropy[w_predicted]
            w_targets = replay_targets[w_predicted]
            if data_ids is not None:
                w_data_ids = replay_data_ids[w_predicted]

            sorted_w = torch.argsort(w_entropy)[:how_much_swap]

            selected_w_idxs = w_idxs[sorted_w]
            selected_w_targets = w_targets[sorted_w]
            if data_ids is not None:
                selected_w_data_ids = w_data_ids[sorted_w]

            if len(sorted_w) < how_much_swap:

                r_predicted = (predicts.cpu() == replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                r_idxs = replay_idxs[r_predicted]
                r_entropy = entropy[r_predicted]
                r_targets = replay_targets[r_predicted]
                    
                if data_ids is not None:
                    r_data_ids = replay_data_ids[r_predicted]

                sorted_r = torch.argsort(r_entropy)[-(how_much_swap-len(sorted_w)):]

                selected_r_idxs = r_idxs[sorted_r]
                selected_r_targets = r_targets[sorted_r]
                    
                if data_ids is not None:
                    selected_r_data_ids = r_data_ids[sorted_r]

                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)

            else:
                selected_idxs = selected_w_idxs
                selected_targets = selected_w_targets
                if data_ids is not None:
                    selected_data_ids = selected_w_data_ids
        
        else:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        assert len(selected_idxs) == how_much_swap
        if diverse: 
            num_regions = 5
            how_much_per_region = int(how_much_swap/num_regions)
            while how_much_per_region <1: 
                num_regions -= 1
                how_much_per_region = int(how_much_swap/num_regions)
                if num_regions == 1: break
            reminder = [1]*int((how_much_swap%num_regions))+[0]*(num_regions-int((how_much_swap%num_regions)))
            k, m = divmod(len(selected_idxs), num_regions)
            selected_idx_partitioned = list((selected_idxs[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(num_regions)))
            selected_idx = []
            for part in selected_idx_partitioned: 
                selected_idx.extend( np.random.choice(len(part), how_much_per_region+reminder.pop(0), replace=False))

        if data_ids is not None:
            return selected_idxs, selected_targets, selected_data_ids
        else:
            return selected_idxs, selected_targets
    def hybrid_easy(self, idxs, outputs, targets, filenames=None, data_ids=None, mode='default'):
        assert('balanced' not in self.swap_options)
        #initialize parameters 
        if mode == 'default':
            swap_ratio = self.threshold
        else:
            swap_ratio = self.threshold2
        if "diverse" in self.swap_options:
            diverse = True 
        else: 
            diverse = False 

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len[0]).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))

        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        replay_filenames = [filenames[i] for i in replay_index_of_idxs]
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        soft_output = self.softmax(replay_output)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        
        if replay_output.nelement() != 0:
            predicts = torch.max(replay_output, dim=1)[1]
            r_predicted = (predicts.cpu() == replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]

            r_idxs = replay_idxs[r_predicted]
            r_entropy = entropy[r_predicted]
            r_targets = replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = replay_data_ids[r_predicted]

            sorted_r = torch.argsort(r_entropy)[:how_much_swap]

            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]

            if len(sorted_r) < how_much_swap:

                w_predicted = (predicts.cpu() != replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                w_idxs = replay_idxs[w_predicted]
                w_entropy = entropy[w_predicted]
                w_targets = replay_targets[w_predicted]
                    
                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]

                sorted_w = torch.argsort(w_entropy)[-(how_much_swap-len(sorted_r)):]

                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]

                selected_idxs = torch.cat((selected_w_idxs,selected_r_idxs),dim=-1)
                selected_targets = torch.cat((selected_w_targets,selected_r_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_w_data_ids,selected_r_data_ids),dim=-1)

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids
        
        else:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        assert len(selected_idxs) == how_much_swap
        if diverse: 
            num_regions = 5
            how_much_per_region = int(how_much_swap/num_regions)
            while how_much_per_region <1: 
                num_regions -= 1
                how_much_per_region = int(how_much_swap/num_regions)
                if num_regions == 1: break
            reminder = [1]*int((how_much_swap%num_regions))+[0]*(num_regions-int((how_much_swap%num_regions)))
            k, m = divmod(len(selected_idxs), num_regions)
            selected_idx_partitioned = list((selected_idxs[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(num_regions)))
            selected_idxs = []
            for part in selected_idx_partitioned: 
                selected_idxs.extend( np.random.choice(len(part), how_much_per_region+reminder.pop(0), replace=False))
            
        if data_ids is not None:
            return selected_idxs, selected_targets, selected_data_ids
        else:
            return torch.tensor(selected_idxs), torch.tensor(selected_targets)

    def hybrid_ratio(self, idxs, outputs, targets, data_ids=None):
        
        swap_ratio = self.threshold

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len[0]).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))

        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        soft_output = self.softmax(replay_output)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        
        if replay_output.nelement() != 0:
            predicts = torch.max(replay_output, dim=1)[1]
            r_predicted = (predicts.cpu() == replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]

            r_idxs = replay_idxs[r_predicted]
            r_entropy = entropy[r_predicted]
            r_targets = replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = replay_data_ids[r_predicted]

            sorted_r = torch.argsort(r_entropy)[:how_much_swap]

            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]

            if len(sorted_r) < how_much_swap:

                w_predicted = (predicts.cpu() != replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                w_idxs = replay_idxs[w_predicted]
                w_entropy = entropy[w_predicted]
                w_targets = replay_targets[w_predicted]
                    
                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]
                
                w_how_much_swap = how_much_swap-len(sorted_r)
                sorted_w = torch.argsort(w_entropy, descending=True)[:w_how_much_swap]

                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]

                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids
        
        else:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        assert len(selected_idxs) == how_much_swap

        if data_ids is not None:
            return selected_idxs, selected_targets, selected_data_ids
        else:
            return selected_idxs, selected_targets

    def get_entropy(self, outputs, targets):
        
        if self._get_entropy == False:
            return

        soft_output = self.softmax(outputs)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        #
        # if wrong predicted sample with low entropy, don't make it swap (make swap FALSE)
        #
        predicts = torch.max(outputs, dim=1)[1]
        r_predicted = (predicts.cpu() == targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        r_entropy = entropy[r_predicted]
            
        self.data_correct_entropy.extend(r_entropy.tolist())

        w_predicted = (predicts.cpu() != targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        w_entropy = entropy[w_predicted]
            
        self.data_wrong_entropy.extend(w_entropy.tolist())


    def get_loss(self, outputs, targets):

        if self._get_loss == False:
            return
        
        try:
            loss = self.swap_loss(outputs, targets)
        except ValueError:
            targets_one_hot = self.to_onehot(targets, outputs.shape[1])
            loss = self.swap_loss(outputs, targets_one_hot)

            loss = loss.view(loss.size(0), -1)
            loss = loss.mean(-1)
        #
        # if wrong predicted sample with low entropy, don't make it swap (make swap FALSE)
        #
        predicts = torch.max(outputs, dim=1)[1]

        r_predicted = (predicts.cpu() == targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        r_loss = loss[r_predicted]
            
        self.data_correct_loss.extend(r_loss.tolist())

        w_predicted = (predicts.cpu() != targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        w_loss = loss[w_predicted]
            
        self.data_wrong_loss.extend(w_loss.tolist())


    def hybrid_loss(self, idxs, outputs, targets, data_ids=None):
        
        swap_ratio = self.threshold

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len[0]).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        total_how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))
        #get the number of each class inside the batch
        batch_dist = {}
        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        
        self.get_loss(replay_output, replay_targets)
        self.get_entropy(replay_output, replay_targets)
        

        for cls in replay_targets:
            if cls.item() not in batch_dist:
                batch_dist[cls.item()] = 1
            else:
                batch_dist[cls.item()] += 1

        expected = 0
        for key in batch_dist.keys():
            batch_dist[key] = math.modf( batch_dist[key] * swap_ratio )
            expected += int(batch_dist[key][1])

        shortage = total_how_much_swap - expected

        if shortage > 0:
            get_dec_samples = list(filter(lambda x: x[1][0] != 0, batch_dist.items()))
            selected_dec_samples = random.sample(get_dec_samples, shortage)
            
            for t in selected_dec_samples:
                k, _ = t
                batch_dist[k] = tuple((batch_dist[k][0] ,batch_dist[k][1] + 1))
        
        if replay_output.nelement() == 0:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        for i, (k,v) in enumerate(batch_dist.items()):
            how_much_swap = int(v[1])

            cur_cls_idx = (replay_targets==k).squeeze().nonzero(as_tuple=True)[0]
            
            cur_replay_output = replay_output[cur_cls_idx]
            cur_replay_idxs = replay_idxs[cur_cls_idx]
            cur_replay_targets = replay_targets[cur_cls_idx]            
            if data_ids is not None:
                cur_replay_data_ids = replay_data_ids[cur_cls_idx]
            
            try:
                loss = self.swap_loss(outputs, targets).cpu()
            except ValueError:
                targets_one_hot = self.to_onehot(targets, outputs.shape[1])
                loss = self.swap_loss(outputs, targets_one_hot).cpu()
                loss = loss.view(loss.size(0), -1)
                loss = loss.mean(-1)
            
            predicts = torch.max(cur_replay_output, dim=1)[1]
            r_predicted = (predicts.cpu() == cur_replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]

            r_idxs = cur_replay_idxs[r_predicted]
            r_loss = loss[r_predicted]            
            r_targets = cur_replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = cur_replay_data_ids[r_predicted]

            sorted_r = torch.argsort(r_loss)[:how_much_swap]

            #to check this code validate
            selected_r_loss = r_loss[sorted_r]

            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]

            if len(sorted_r) < how_much_swap:
                w_predicted = (predicts.cpu() != cur_replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                w_idxs = cur_replay_idxs[w_predicted]
                w_loss = loss[w_predicted]

                w_targets = cur_replay_targets[w_predicted]
                    
                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]

                w_how_much_swap = how_much_swap-len(sorted_r)
                sorted_w = torch.argsort(w_loss)[:w_how_much_swap]

                #to check this code validate
                selected_w_loss = w_loss[sorted_w]

                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]

                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids
        
            if i==0:                
                total_selected_idxs = selected_idxs
                total_selected_targets = selected_targets
                if data_ids is not None:
                    total_selected_data_ids = selected_data_ids

            else:
                total_selected_idxs = torch.cat((total_selected_idxs,selected_idxs),dim=-1)
                total_selected_targets = torch.cat((total_selected_targets,selected_targets),dim=-1)
                if data_ids is not None:
                    total_selected_data_ids = torch.cat((total_selected_data_ids,selected_data_ids),dim=-1)

        
        assert len(total_selected_idxs) == total_how_much_swap

        if data_ids is not None:
            return total_selected_idxs, total_selected_targets, total_selected_data_ids
        else:
            return total_selected_idxs, total_selected_targets

    ###########################################################################
    
    def gradient_based(self, idxs, outputs, targets, avg_grad, avg_var, data_ids=None,mode='default'):
        # which samples to swap out
        if "hard" in self.swap_options:
            reverse = True 
        elif "easy" in self.swap_options:
            reverse = False 
        else: reverse = False
        # whether diversity is considered or not 
        if "diverse" in self.swap_options:
            diverse = True 
        else: 
            diverse = False 
        # whether we swap out the same ratio for each class
        if "balanced" in self.swap_options:
            balanced= True 
        else: 
            balanced = False 

        assert(not(diverse==True and balanced==True)) # diverse and balanced are mutually exclusive
        if balanced and diverse==False:  # Class balanced 
            return self.gradtail_balanced(idxs, outputs, targets, avg_grad, avg_var)
        
        elif balanced==False: # not class balanced 
            return self.gradtail(idxs, outputs, targets, avg_grad, avg_var)
            
        pass
    
    def _gradtail(self, llist, xlist, avg_grad, avg_var):
        g_device = xlist.device
        g_lossl = ((llist * ((-1 * llist) + 1)) * xlist) 
        l_x, l_y = g_lossl.shape[0], g_lossl.shape[1]
        l_pre = avg_grad.shape[0]
        
        if l_x > l_pre:
            avg_add = torch.ones([(l_x - l_pre), 1], dtype=torch.float64, device=g_device, requires_grad=True)
            avg_grad = torch.cat([avg_grad, (avg_add * avg_grad.mean())], dim=0)
            avg_var = torch.cat([avg_var, (avg_add * avg_var.mean())], dim=0)
        g_avg = (torch.mean(g_lossl, dim=1, keepdim=True))#.repeat(1, l_y)
        g_lambda = 0.99
        avg_grad_i = avg_grad[0:l_x]
        g_dot = g_lossl * ((torch.div(avg_grad_i, (avg_grad_i + 1))).repeat(1, l_y))
        avg_grad_t= (g_avg * (1-g_lambda)) + (avg_grad_i * (g_lambda)) # * g_thetamid
        for i in range(l_x): avg_grad[i, 0] = avg_grad_t[i, 0]
        
        g_min, _ = torch.min(g_dot, dim=1, keepdim=True)
        g_max, _ = torch.max(g_dot, dim=1, keepdim=True)
        g_range = (g_max - g_min).repeat(1, l_y)
        
        if ("easy" in self.swap_options) or ("hard_f" in self.swap_options):
            # easy = 1: high at close to 1
            g_threshold = -1
            g_easy_hard = 1
        elif "hard" in self.swap_options:
            # hard = -1: high at close to -1
            g_threshold = 1
            g_easy_hard = -1
        elif "rare" in self.swap_options:
            # rare = 0: high at close to 0
            g_threshold = 0
            g_easy_hard = 1
        else: 
            # same with shift easy: if sample is easy, g_loss is high
            g_threshold = -1
            g_easy_hard = 1
            
        g_theta = torch.div(((g_dot) - g_min.repeat(1, l_y)), g_range)
        g_theta = (g_theta * 2) - 1

        # Remove the abs(), choose positive ones (easy samples)
        g_var = torch.mean(torch.abs(g_theta), dim=1, keepdim=True)#.repeat(1, l_y)
        avg_var_i = (g_var * (1-g_lambda)) + (avg_var[0:l_x] * (g_lambda))
        for j in range(l_x): avg_var[j, 0] = avg_var_i[j, 0]
        self.avg_gradt = avg_grad
        self.avg_vart = avg_var
        
        g_input = torch.div(g_theta, avg_var_i.repeat(1, l_y))
        g_weight = (g_input + g_threshold) * g_easy_hard
        
        g_func = torch.abs(1 / (1 - torch.exp(g_weight))) + 1
        g_lossn = g_func * g_lossl 
        g_loss = (torch.sum(g_lossn, 1)).squeeze()        

        return g_loss 
    
    def gradtail(self, idxs, outputs, targets, avg_grad, avg_var, data_ids=None,mode='default'):
        
        if mode == 'default':
            swap_ratio = self.threshold
        else:
            swap_ratio = self.threshold2
            
        # which samples to swap out
        if "easy" in self.swap_options:
            reverse = True 
        elif "hard" in self.swap_options:
            reverse = False 
        else: reverse = False
        
        # whether diversity is considered or not 
        if "diverse" in self.swap_options:
            diverse = True 
        else: 
            diverse = False 

        if swap_ratio == 0: return torch.tensor([]),torch.tensor([])
        
        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len[0]).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)
        
        soft_output = self.softmax(outputs)
        gradient = self._gradtail(soft_output, outputs, avg_grad, avg_var)
        
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        replay_gradient = (gradient.cpu())[replay_index_of_idxs].clone().detach()
        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))
        
        # Choose more easy 
        if reverse == False:
            selected_idx = sorted(range(len(replay_gradient)), key=replay_gradient.__getitem__ ,reverse=True)[:how_much_swap]

        else: # choose more hard
            selected_idx = sorted(range(len(replay_gradient)), key=replay_gradient.__getitem__)[:how_much_swap]
        
        if diverse: 
            num_regions = 5
            how_much_per_region = int(how_much_swap/num_regions)
            while how_much_per_region <1: 
                num_regions -= 1 
                how_much_per_region = int(how_much_swap/num_regions)
                if num_regions == 1: break
            reminder = [1]*int((how_much_swap%num_regions))+[0]*(num_regions-int((how_much_swap%num_regions)))
            k, m = divmod(len(selected_idx), num_regions)
            selected_idx_partitioned = list((selected_idx[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(num_regions)))
            selected_idx = []
            for part in selected_idx_partitioned: 
                selected_idx.extend( np.random.choice(len(part), how_much_per_region+reminder.pop(0), replace=False))
            selected_idx = selected_idx
        
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()
            swap_idx_en, swap_target_en, swap_data_ids = replay_idxs[selected_idx], replay_targets[selected_idx], replay_data_ids[selected_idx]
            return swap_idx_en, swap_target_en, self.avg_gradt, self.avg_vart, swap_data_ids
        else:
            swap_idx_en, swap_target_en = replay_idxs[selected_idx], replay_targets[selected_idx]
            return swap_idx_en, swap_target_en, self.avg_gradt, self.avg_vart
        
    def gradtail_balanced(self, idxs, outputs, targets, avg_grad, avg_var, data_ids=None,mode='default'):
        assert(not "diverse" in self.swap_options)

        if mode == 'default':
            swap_ratio = self.threshold
        else:
            swap_ratio = self.threshold2
            
        # which samples to swap out
        if "easy" in self.swap_options:
            reverse = True 
        elif "hard" in self.swap_options:
            reverse = False 
        else: reverse = False

        if swap_ratio == 0: return torch.tensor([]),torch.tensor([])
        
        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len[0]).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)
        
        soft_output = self.softmax(outputs)
        gradient = self._gradtail(soft_output, outputs, avg_grad, avg_var)
        
        # replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        replay_gradient = gradient[replay_index_of_idxs].clone().detach()
        total_how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))
                        
        batch_dist = {}
        for cls in replay_targets:
            if cls.item() not in batch_dist:
                batch_dist[cls.item()] = 1
            else:
                batch_dist[cls.item()] += 1

        expected = 0
        for key in batch_dist.keys():
            batch_dist[key] = math.modf( batch_dist[key] * swap_ratio )
            expected += int(batch_dist[key][1])

        shortage = total_how_much_swap - expected
        if shortage > 0:
            get_dec_samples = list(filter(lambda x: x[1][0] != 0, batch_dist.items()))
            selected_dec_samples = random.sample(get_dec_samples, shortage)
            
            for t in selected_dec_samples:
                k, _ = t
                batch_dist[k] = tuple((batch_dist[k][0] ,batch_dist[k][1] + 1))
        
        total_selected_idx=[]
        #separate batch into class-wise
        for i, (k,v) in enumerate(batch_dist.items()):

            how_much_swap = int(v[1])
            if how_much_swap == 0:
                continue
            cur_cls_idx = (replay_targets==k).squeeze().nonzero(as_tuple=True)[0]
            
            if (len(cur_cls_idx) <= 1) or (len(cur_cls_idx) == how_much_swap): 
                total_selected_idx.extend(cur_cls_idx.tolist())
                continue
            
            assert(len(cur_cls_idx) > how_much_swap)

            cur_replay_gradient = (replay_gradient[cur_cls_idx]).tolist()

            # Choose more easy 
            if reverse == False:
                cur_idx = sorted(range(len(cur_replay_gradient)), key=cur_replay_gradient.__getitem__ ,reverse=True)
                
            else: # choose more hard
                cur_idx = sorted(range(len(cur_replay_gradient)), key=cur_replay_gradient.__getitem__)
            
            selected_idx = [(cur_cls_idx.tolist())[cur_idx[i]] for i in range(how_much_swap)]
            total_selected_idx.extend(selected_idx)
        pass
            
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()
            swap_idx_en, swap_target_en, swap_data_ids = replay_idxs[total_selected_idx], replay_targets[total_selected_idx], replay_data_ids[total_selected_idx]
            return swap_idx_en, swap_target_en, self.avg_gradt, self.avg_vart, swap_data_ids
        else:
            swap_idx_en, swap_target_en = replay_idxs[total_selected_idx], replay_targets[total_selected_idx]
            return swap_idx_en, swap_target_en, self.avg_gradt, self.avg_vart
    
    ###########################################################################
     

    
    #@profile
    def all(self, idxs, outputs=None, targets=None, data_ids=None):
    
        ######### changed for time measurement
        if targets is not None and data_ids is not None:   
            return self.get_replay_index(idxs, targets, data_ids)
        else:
            return self.get_replay_index(idxs, targets)
            

    def hybrid_random(self, idxs, outputs, targets, data_ids=None):
        
        swap_ratio = self.threshold

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len[0]).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))

        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        soft_output = self.softmax(replay_output)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        
        if replay_output.nelement() != 0:
            predicts = torch.max(replay_output, dim=1)[1]
            r_predicted = (predicts.cpu() == replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]

            r_idxs = replay_idxs[r_predicted]
            r_entropy = entropy[r_predicted]
            r_targets = replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = replay_data_ids[r_predicted]

            
            sorted_r = torch.argsort(r_entropy)[:how_much_swap]

            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]


            if len(sorted_r) < how_much_swap:

                w_predicted = (predicts.cpu() != replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                w_idxs = replay_idxs[w_predicted]
                w_entropy = entropy[w_predicted]
                w_targets = replay_targets[w_predicted]
                    
                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]
                
                w_how_much_swap = how_much_swap-len(sorted_r)

                idx_to_pick = np.linspace(0, len(w_idxs), num=w_how_much_swap, dtype = int, endpoint=False)
                sorted_w = torch.argsort(w_entropy, descending=True)[idx_to_pick]

                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]

                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids
        
        else:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        assert len(selected_idxs) == how_much_swap


        if data_ids is not None:
            return selected_idxs, selected_targets, selected_data_ids
        else:
            return selected_idxs, selected_targets



def fetch_from_storage(replay_dataset, swap_idxs,labels,task_id,filenames=None,testset='urbansound8k',transform=None):
    for i in range(len(swap_idxs)):
        idx = swap_idxs[i]
        label = labels[i]
        path_curr_label = replay_dataset.rb_path + '/' + str(label)
        try:
            prefix = f'{label}_'
            if testset in ['audioset']:
                suffix ='.npy'
            else:
                suffix = '.png'
            if testset == 'urbansound8k':
                num_file = US8K_INDEX[label]
            elif testset == 'imagenet1000':
                num_file = IMG1000_INDEX[label]
            elif testset == 'dailynsports':
                num_file = DSADS_INDEX[label]
            else: num_file=500
            if testset == 'dailynsports':
                if filenames ==None:
                    replace_file = path_curr_label + '/' + prefix+str(random.randint(1,num_file)) + '.npy'
                else: 
                    file_no = eval(filenames[i].split('_')[-1])
                    if file_no > num_file:replace_file = path_curr_label + '/' + prefix+str(random.randint(1,num_file)) + '.npy'
                    else: replace_file = path_curr_label + '/' +filenames[i] + '.npy'
            else:
                if filenames ==None:
                    replace_file = path_curr_label + '/' + prefix+str(random.randint(1,num_file)) + '.png'
                else: 
                    file_no = eval(filenames[i].split('_')[-1])
                    if file_no > num_file:replace_file = path_curr_label + '/' + prefix+str(random.randint(1,num_file)) + '.png'
                    else: replace_file = path_curr_label + '/' +filenames[i] + '.png'
        except Exception as e:
            print(f'swap_manager.fetch_from_storage: {e}')
            return False
        get_file(idx, replace_file,replay_dataset,transform=transform)
def get_file(idx, filename, replay_dataset,transform=None):
    if 'png' in filename:
        vec = _get_img(filename)
    else:
        vec = np.load(filename)
        
    swap_in_file = filename.split('/')[-1]
    swap_in_file = swap_in_file.split('.png')[0]

    data_at_idx = shared_memory.SharedMemory(name=replay_dataset.data_shm_name[idx])
    if transform is not None:
        vec = np.array( transform(vec))
    data = np.ndarray(replay_dataset.vec_shape,replay_dataset.vec_dtype,buffer=data_at_idx.buf)
    data[:] = vec
    replay_dataset.filenames[idx] = swap_in_file 
    return True

def _get_img(filename):
    f = os.open( filename, os.O_RDONLY | os.O_DIRECT)

    os.lseek(f,0,0)
    actual_size = os.path.getsize(filename)
    block_size = 512 * math.ceil(actual_size / 512)
    fr = directio.read(f, block_size)
    os.close(f)
    
    data = io.BytesIO(fr[:actual_size])
    

    img = Image.open(data)
    img = img.convert('RGB')
    
    return img


class _DatasetSwapper_ImageNet1k(object):
    def __init__(self, dataset, saver, testset='cifar100',task_id=0):
        print("[Init Swapper]")
        if hasattr(dataset, 'replay_dataset'):
            self.dataset = dataset.replay_dataset
        else:
            self.dataset = dataset
        self.saver = saver
        self.testset=testset
        if testset in ["twentynews", "dailynsports", "shakespeare","audioset"]: self.suffix = '.npy'
        else: self.suffix = '.png'
    async def _get_logit(self, logit_filename):
        f = os.open( logit_filename, os.O_RDONLY | os.O_DIRECT)
        os.lseek(f,0,0)
        actual_size = os.path.getsize(logit_filename)
        block_size = 512 * math.ceil(actual_size / 512)
        fr = directio.read(f, block_size)
        os.close(f)
        
        data = io.BytesIO(fr[:actual_size])
        
        logit = pickle.load(data)

        return logit

    async def _get_img(self, filename):
        f = os.open( filename, os.O_RDONLY | os.O_DIRECT)

        os.lseek(f,0,0)
        actual_size = os.path.getsize(filename)
        block_size = 512 * math.ceil(actual_size / 512)
        try:
            fr = directio.read(f, block_size)
        except Exception:
            print(f'Failed reading {filename}',flush=True) 
        os.close(f)
        
        data = io.BytesIO(fr[:actual_size])
        

        img = Image.open(data)
        img = img.convert('RGB')
        
        return img
    
    
    async def _get_file_list(self, path):
        list_dir = os.listdir(path)
        return list_dir

    async def _get_data(self, idx, filename, data_id=None):

        if 'png' in filename or 'JPEG' in filename:
            vec = await self._get_img(filename)
        else:
            vec = np.load(filename)
        swap_in_file = filename.split('/')[-1]
        swap_in_file = swap_in_file.split(self.suffix)[0]
        self.dataset.data[idx] = vec
        self.dataset.filenames[idx] = swap_in_file
        return True

    async def _swap_main(self, label, swap_idx, swap_in_file=None, data_id=None):
        if swap_in_file is None: #str
            path_curr_label = self.dataset.rb_path + '/' + str(label)
        else:
            path_curr_label = self.dataset.rb_path + '/' + str(label)
        # try:
        if swap_in_file is None:
            prefix = f'{label}_'
            if self.saver is None : 
                num_file = IMG1000_INDEX[label]
            else: num_file = self.saver.get_num_file_for_label_for_swap(label)
            replace_file = path_curr_label + '/' + prefix+str(random.randint(1,num_file)) + self.suffix
        else: 
            replace_file = path_curr_label + '/' + swap_in_file + self.suffix
        return await self._get_data(swap_idx, replace_file, data_id)
    def update_meta(self,rp_len):
        self.rp_len=rp_len
    async def _swap(self, what_to_swap, labels, swap_in_file=None, data_ids=None):
        cos = [ self._swap_main(label, idx ) for label, idx in zip(labels, what_to_swap) ]
        res = await asyncio.gather(*cos)
        return res
class SwapManager_ImageNet1k(object):
    total_count = 0
    overlap_count = 0
    manager = python_multiprocessing.Manager()

    def __init__(self, replay_dataset, num_workers, swap_base, store_budget=None, get_loss=False, get_entropy=False, seed=None, **kwargs):
        self.swap_base = swap_base
        self.swap_determine = self.swap_policy(swap_base)
        self.softmax = torch.nn.Softmax(dim=1)
        self.num_workers = num_workers
        self.dataset = replay_dataset
        self.num_swap = 0
        self.cls_num_swap = dict()
        self.agent = self.dataset.agent
        self._swap_loss = None
        self.new_classes = None
        self.seed = seed

        self.swap_class_dist = {}
        print(f'Swapper Parameters: {kwargs}')
        if 'dataset' in kwargs:
            if kwargs['dataset'] in ['imagenet1000', 'imagenet100','miniimagenet']:
                self.test_set =kwargs['dataset'] 
            elif kwargs['dataset'] in ["twentynews", "dailynsports", "shakespeare"]:
                self.test_set = kwargs['dataset'] 

        if 'saver' in kwargs and kwargs['saver'] == False:
            self.saver = None 
            print('[Saver Disabled]')
        else:
            self.saver = DataSaver(replay_dataset.rb_path, store_budget, seed, self.test_set)
            print('[Creating Saver]')
            print(f'Saver Path: {self.saver.rb_path}')
        if num_workers == 0:
            self._swapper = _DatasetSwapper_ImageNet1k(self.dataset, self.saver,self.test_set)

        if 'threshold' in kwargs:
            self.threshold = float(kwargs['threshold'][0])
            if len(kwargs['threshold'])>1:
                self.threshold2 = float(kwargs['threshold'][1])
            else: 
                self.threshold2 = float(kwargs['threshold'][0])
        
        if 'filename' in kwargs:
            self.filename = kwargs['filename']

        if 'result_save_path' in kwargs:
            self.result_save_path = kwargs['result_save_path']
        
        self._get_loss = get_loss
        self._get_entropy = get_entropy

        self.data_correct_entropy = []
        self.data_wrong_entropy = []

        self.data_correct_loss = []
        self.data_wrong_loss = []
        
        self.stream_loss = []
        self.replay_loss = []
    # args=(i, self.dataset, self.saver, swap_queue, self._issue_done_event,self._all_swapped_event,self._reset_n_pause,self._cleared,self.seed, self._num_issued,self.task_id)

    def adaptive_swap_loop(self,worker_id, dataset, saver, swap_queue, all_recved, all_swapped,pause_n_reset,cleared,seed,num_issued,task_id):
        #torch.set_num_threads(1)
        #torch.manual_seed(seed)

        if seed is not None:
            np.random.seed(seed + worker_id)
            torch.manual_seed(seed + worker_id)
            random.seed(seed + worker_id)
            torch.cuda.manual_seed_all(seed + worker_id)
            torch.cuda.manual_seed(seed + worker_id)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        swapper = _DatasetSwapper_ImageNet1k(dataset,saver,testset=self.test_set,task_id=task_id)
        
        while True:
            try:
                idxs,targets,filenames,data_ids = swap_queue.get(block=False)
                
            except queue.Empty: 
                cleared.set()
                time.sleep(1)
                continue
            cleared.clear()
            if pause_n_reset.is_set(): 
                continue
            
            if idxs is None and targets is None:
                assert all_recved.is_set()
                all_swapped.set()
                break
            elif idxs[0] == -7 : 
                self.rp_len = targets
                swapper.update_meta(targets)
                continue            
            elif all_recved.is_set():
                continue
            elif all_swapped.is_set():
                break
            
            swap_idx,labels = self.swap_determine(idxs,targets,filenames,data_ids)
            
            swap_idx, labels = swap_idx.tolist(), labels.tolist()
            
            # Determine swap-in file here, if we need to determine it in advance 
            swap_in_file = None        
            if swap_in_file is not None:
                swap_res = asyncio.run(swapper._swap(swap_idx, labels, swap_in_file))
            else:
                swap_res = asyncio.run(swapper._swap(swap_idx, labels))

            num_issued.value += len(swap_idx)
            del swap_idx, labels, swap_in_file,data_ids    
    def set_swap_ratio(self,coeff):
        self.threshold *= coeff
        self.threshold2 *= coeff
        return [self.threshold,self.threshold2]
    def get_swap_ratio(self):
        return np.min([self.threshold,self.threshold2])
    def before_train(self,dataset=None,task_id=0):  
        self.task_id = task_id
        if dataset is not None: 
            self.dataset = dataset
        if self.agent in ["der","derpp", "tiny","aser"]:
            self.rp_len = None
        else:
            self.rp_len = len(self.dataset)

        if self.saver is not None:
            self.saver.before_train()
        
        if self.num_workers > 0 :
            #swap process
            self._swap_workers=[]
            self._swap_queues=[]
            self._issue_done_event = python_multiprocessing.Event()
            self._reset_n_pause = python_multiprocessing.Event()
            self._cleared = python_multiprocessing.Event()
            self._all_swapped_event = python_multiprocessing.Event()
            self._swap_worker_queue_idx_cycle = itertools.cycle(range(self.num_workers))
            self._num_issued = python_multiprocessing.Value('i',0)
            self._swap_worker_pids=[]
            for i in range(self.num_workers):
                swap_queue = python_multiprocessing.Queue()        
                swap_worker = python_multiprocessing.Process(
                    target = self.adaptive_swap_loop,
                    args=(i, self.dataset, self.saver, swap_queue, self._issue_done_event,self._all_swapped_event,self._reset_n_pause,self._cleared,self.seed, self._num_issued,self.task_id)
                )
                swap_worker.daemon = True
                swap_worker.start()
                
                self._swap_workers.append(swap_worker)
                self._swap_queues.append(swap_queue)
    def pause(self):
        self._reset_n_pause.set()
        while True:
            if self._cleared.is_set():break
            else: time.sleep(1)

    def resume(self):
        self._reset_n_pause.clear()
        while True:
            if self._cleared.is_set():break
            else: time.sleep(0.1)

    def after_train(self, get_loss=False, get_entropy=False,now=False):
        #shudown swap process
        if self.num_workers > 0:
            self._issue_done_event.set()
            if now == True: self._all_swapped_event.set()
            for sq in self._swap_queues:
                sq.put((None, None, None, None))
            
            for s in self._swap_workers:
                s.join() #timeout?

            for sq in self._swap_queues:
                sq.cancel_join_thread()
                sq.close()

            gc.collect()
            print("[Swapper shutdown]")
            if self.saver is not None:
                self.saver.after_train()
    
        self.data_correct_loss = []
        self.data_wrong_loss = []
        self.data_correct_entropy = []
        self.data_wrong_entropy = []
        all_swapped, num_issued =  self._all_swapped_event.is_set(), self._num_issued.value
        #reset events
        self._issue_done_event.clear()
        self._reset_n_pause.clear()
        self._all_swapped_event.clear()
        self._cleared.set()
        return all_swapped, num_issued
    def get_num_swap(self):
        return self.num_swap
    def get_cls_num_swap(self):
        return self.cls_num_swap
    def reset_num_swap(self):
        self.num_swap = 0
        self.cls_num_swap = self.cls_num_swap.fromkeys(self.cls_num_swap,0)
    def reset_swap_class_dist(self):
        self.swap_class_dist = {}

    def swap(self, what_to_swap, labels):

        None_list = [None] * len(what_to_swap)
        if len(what_to_swap) > 0:
            if hasattr(self, "_swapper"):
                asyncio.run(self._swapper._swap(what_to_swap, labels))
            elif hasattr(self, "_swap_queues"):
                worker_id = next(self._swap_worker_queue_idx_cycle)
                self._swap_queues[worker_id].put(
                    (what_to_swap, labels,None_list,None))
            self.num_swap = self.num_swap + len(what_to_swap)
            counter = Counter(labels)
            for key in counter:
                if key not in self.cls_num_swap: self.cls_num_swap[key] = counter[key]
                else: 
                     self.cls_num_swap[key] += counter[key]
               
    def swap_pt(self, idxs,targets,filenames=None,data_ids=None):  
        
        if len(idxs) > 0:
            swap_in_file = [None] * len(idxs)
            filenames= [None] * len(idxs)
            if hasattr(self, "_swapper"):
                asyncio.run(self._swapper._swap_pt(what_to_swap, labels, swap_in_file))
            elif hasattr(self, "_swap_queues"):
                worker_id = next(self._swap_worker_queue_idx_cycle)
                self._swap_queues[worker_id].put(
                    (idxs,targets,filenames,data_ids))
    def update_meta(self, rp_len): 
        self.rp_len=rp_len
        if hasattr(self, "_swapper"):
            asyncio.run(self._swapper._swap_pt([-7], rp_len, None))
        elif hasattr(self, "_swap_queues"):
            worker_id = next(self._swap_worker_queue_idx_cycle)
            self._swap_queues[worker_id].put(
                ([-7],rp_len,None,None)) 
    def swap_policy(self, swap_base):
        policies = {
            "entropy" : self.entropy,
            "hybrid_threshold" : self.hybrid,
            "prediction" : self.prediction,
            "random" : self.random,
            "random_fixed": self.random_fixed,
            "pure_random" : self.pure_random,
            "hybrid_ratio" : self.hybrid_ratio,
            "hybrid_balanced" : self.hybrid_balanced,
            "hybrid_balanced_p" : self.hybrid_balanced_p,
            "hybrid_loss" : self.hybrid_loss,
            "all" : self.all
        }
        return policies[swap_base]

    @property
    def swap_thr(self):
        return self._swap_thr
    
    @swap_thr.setter
    def swap_thr(self, thr):
        self._swap_thr = thr

    
    @property
    def swap_loss(self):
        return self._swap_loss
    
    @swap_loss.setter
    def swap_loss(self, loss):
        self._swap_loss = loss
    
    def to_onehot(self, targets, n_classes):
        onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
        onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
        return onehot

    def get_replay_index(self, idxs, targets, data_ids=None):

        if self.rp_len is None:
            if data_ids is not None:
                return idxs, targets, data_ids
            else:
                return idxs, targets

        else:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
            
            if data_ids is not None:
                return idxs[replay_index_of_idxs], targets[replay_index_of_idxs], data_ids[replay_index_of_idxs]
            else:
                return idxs[replay_index_of_idxs], targets[replay_index_of_idxs]

    def prediction(self, idxs, outputs, targets, data_ids=None):
        #
        # determine what to swap based on mis-prediction
        #
        predicts = torch.max(outputs, dim=1)[1]
        selected_idx = (predicts.cpu() == targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        if data_ids is not None:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx], data_ids[selected_idx])
        else:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx])

    def entropy(self, idxs, outputs, targets, data_ids=None):
        #
        # determine what to swap based on entropy (threshold = 1.0 : lower is easy and swap, higher is hard and preserve)
        #
        soft_output = self.softmax(outputs)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        selected_idx = (entropy.cpu() < self.threshold).squeeze().nonzero(as_tuple=True)[0]

        if data_ids is not None:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx], data_ids[selected_idx])
        else:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx])


    def hybrid(self, idxs, outputs, targets, data_ids=None):
        #
        # determine what to swap based on entropy (threshold : lower is easy and swap, higher is hard and preserve)
        #        
        soft_output = self.softmax(outputs)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        entropy_batch = (entropy.cpu() < self.threshold).squeeze()
        #
        # if wrong predicted sample with low entropy, don't make it swap (make swap FALSE)
        #
        predicts = torch.max(outputs, dim=1)[1]
        prediction_batch = (predicts.cpu() == targets.cpu()).squeeze()

        selected_idx = (torch.logical_and(entropy_batch, prediction_batch)).nonzero(as_tuple=True)[0]
        if data_ids is not None:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx], data_ids[selected_idx])
        else:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx])
    
    def random_fixed(self, idxs, outputs, targets, data_ids=None):
        
        swap_ratio = self.threshold
        
        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)
        
        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))
        
        replay_output = outputs[replay_index_of_idxs]
        replay_idxs = idxs[replay_index_of_idxs]
        replay_targets = targets[replay_index_of_idxs]
        
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs]
        
        if data_ids is not None:
            return replay_idxs[:how_much_swap], replay_targets[:how_much_swap], replay_data_ids[:how_much_swap]
        else:
            return replay_idxs[:how_much_swap], replay_targets[:how_much_swap]
            

    def random(self, idxs, targets, filenames, data_ids=None):
        
        swap_ratio = self.threshold
        # idxs and targets are lists now 
        idxs, targets = torch.tensor(idxs),torch.tensor(targets)
        if swap_ratio == 0: return torch.tensor([]),torch.tensor([])
        
        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)
        
        
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if (self.threshold==1): return replay_idxs, replay_targets
        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))
        selected_index = np.random.choice(len(replay_idxs), how_much_swap, replace=False)
        assert len(selected_index) == how_much_swap

        return replay_idxs[selected_index], replay_targets[selected_index] #, outputs[selected_index]
    
    def pure_random(self, idxs, outputs, targets, data_ids=None):

        swap_ratio = self.threshold

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))

        idx_to_pick = np.linspace(0, len(replay_index_of_idxs), num=how_much_swap, dtype = int, endpoint=False)
        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        
        soft_output = self.softmax(replay_output)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()

        if replay_output.nelement() != 0:
            predicts = torch.max(replay_output, dim=1)[1]
            r_predicted = (predicts.cpu() == replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]

            r_idxs = replay_idxs[r_predicted]
            r_entropy = entropy[r_predicted]
            r_targets = replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = replay_data_ids[r_predicted]
    
            r_range = np.where(idx_to_pick < len(r_idxs))
            sorted_r = torch.argsort(r_entropy)[idx_to_pick[r_range]]

            
            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]


            if len(sorted_r) < how_much_swap:
                idx_to_pick = np.delete(idx_to_pick, r_range)
                idx_to_pick = idx_to_pick - len(r_idxs)

                w_predicted = (predicts.cpu() != replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                
                w_idxs = replay_idxs[w_predicted]
                w_entropy = entropy[w_predicted]
                w_targets = replay_targets[w_predicted]

                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]
                
                sorted_w = torch.argsort(w_entropy, descending=True)[idx_to_pick]


                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]


                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)
        

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids


        else:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        assert len(selected_idxs) == how_much_swap

        if data_ids is not None:
            return selected_idxs, selected_targets, selected_data_ids
        else:
            return selected_idxs, selected_targets


    def hybrid_opposite(self, idxs, outputs, targets, data_ids=None):

        swap_ratio = self.threshold

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))

        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        soft_output = self.softmax(replay_output)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        
        if replay_output.nelement() != 0:
            predicts = torch.max(replay_output, dim=1)[1]
            w_predicted = (predicts.cpu() != replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]

            w_idxs = replay_idxs[w_predicted]
            w_entropy = entropy[w_predicted]
            w_targets = replay_targets[w_predicted]
            if data_ids is not None:
                w_data_ids = replay_data_ids[w_predicted]

            
            sorted_w = torch.argsort(w_entropy)[:how_much_swap]

            selected_w_idxs = w_idxs[sorted_w]
            selected_w_targets = w_targets[sorted_w]
            if data_ids is not None:
                selected_w_data_ids = w_data_ids[sorted_w]

            if len(sorted_w) < how_much_swap:

                r_predicted = (predicts.cpu() == replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                r_idxs = replay_idxs[r_predicted]
                r_entropy = entropy[r_predicted]
                r_targets = replay_targets[r_predicted]
                    
                if data_ids is not None:
                    r_data_ids = replay_data_ids[r_predicted]

                sorted_r = torch.argsort(r_entropy)[-(how_much_swap-len(sorted_w)):]

                selected_r_idxs = r_idxs[sorted_r]
                selected_r_targets = r_targets[sorted_r]
                    
                if data_ids is not None:
                    selected_r_data_ids = r_data_ids[sorted_r]
                    
                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)

            else:
                selected_idxs = selected_w_idxs
                selected_targets = selected_w_targets
                if data_ids is not None:
                    selected_data_ids = selected_w_data_ids
        
        else:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        assert len(selected_idxs) == how_much_swap


        if data_ids is not None:
            return selected_idxs, selected_targets, selected_data_ids
        else:
            return selected_idxs, selected_targets

    def hybrid_ratio(self, idxs, outputs, targets, data_ids=None):
        
        swap_ratio = self.threshold

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))

        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        soft_output = self.softmax(replay_output)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        if replay_output.nelement() != 0:
            predicts = torch.max(replay_output, dim=1)[1]
            r_predicted = (predicts.cpu() == replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]

            r_idxs = replay_idxs[r_predicted]
            r_entropy = entropy[r_predicted]
            r_targets = replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = replay_data_ids[r_predicted]

            sorted_r = torch.argsort(r_entropy)[:how_much_swap]

            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]

            if len(sorted_r) < how_much_swap:

                w_predicted = (predicts.cpu() != replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                w_idxs = replay_idxs[w_predicted]
                w_entropy = entropy[w_predicted]
                w_targets = replay_targets[w_predicted]
                    
                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]

                w_how_much_swap = how_much_swap-len(sorted_r)
                sorted_w = torch.argsort(w_entropy, descending=True)[:w_how_much_swap]

                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]

                    
                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids
        
        else:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        assert len(selected_idxs) == how_much_swap

        if data_ids is not None:
            return selected_idxs, selected_targets, selected_data_ids
        else:
            return selected_idxs, selected_targets

    def get_entropy(self, outputs, targets):
        
        if self._get_entropy == False:
            return
        soft_output = self.softmax(outputs)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        #
        # if wrong predicted sample with low entropy, don't make it swap (make swap FALSE)
        #
        predicts = torch.max(outputs, dim=1)[1]
        r_predicted = (predicts.cpu() == targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        r_entropy = entropy[r_predicted]
            
        self.data_correct_entropy.extend(r_entropy.tolist())

        w_predicted = (predicts.cpu() != targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        w_entropy = entropy[w_predicted]
            
        self.data_wrong_entropy.extend(w_entropy.tolist())


    def get_loss(self, outputs, targets):

        if self._get_loss == False:
            return

        try:
            loss = self.swap_loss(outputs, targets)
        except ValueError:
            targets_one_hot = self.to_onehot(targets, outputs.shape[1])
            loss = self.swap_loss(outputs, targets_one_hot)

            loss = loss.view(loss.size(0), -1)
            loss = loss.mean(-1)
        #
        # if wrong predicted sample with low entropy, don't make it swap (make swap FALSE)
        #
        predicts = torch.max(outputs, dim=1)[1]

        r_predicted = (predicts.cpu() == targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        r_loss = loss[r_predicted]
            
        self.data_correct_loss.extend(r_loss.tolist())

        w_predicted = (predicts.cpu() != targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        w_loss = loss[w_predicted]
            
        self.data_wrong_loss.extend(w_loss.tolist())


    def hybrid_loss(self, idxs, outputs, targets, data_ids=None):
        
        swap_ratio = self.threshold

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        total_how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))

        #get the number of each class inside the batch
        batch_dist = {}
        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        
        self.get_loss(replay_output, replay_targets)
        self.get_entropy(replay_output, replay_targets)
        

        for cls in replay_targets:
            if cls.item() not in batch_dist:
                batch_dist[cls.item()] = 1
            else:
                batch_dist[cls.item()] += 1

        expected = 0
        for key in batch_dist.keys():
            batch_dist[key] = math.modf( batch_dist[key] * swap_ratio )
            expected += int(batch_dist[key][1])

        shortage = total_how_much_swap - expected

        if shortage > 0:
            get_dec_samples = list(filter(lambda x: x[1][0] != 0, batch_dist.items()))
            selected_dec_samples = random.sample(get_dec_samples, shortage)
            
            for t in selected_dec_samples:
                k, _ = t
                batch_dist[k] = tuple((batch_dist[k][0] ,batch_dist[k][1] + 1))
        
        if replay_output.nelement() == 0:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)
            
        #separate batch into class-wise
        for i, (k,v) in enumerate(batch_dist.items()):
            how_much_swap = int(v[1])

            cur_cls_idx = (replay_targets==k).squeeze().nonzero(as_tuple=True)[0]
            
            cur_replay_output = replay_output[cur_cls_idx]
            cur_replay_idxs = replay_idxs[cur_cls_idx]
            cur_replay_targets = replay_targets[cur_cls_idx]            
            if data_ids is not None:
                cur_replay_data_ids = replay_data_ids[cur_cls_idx]
            
            
            try:
                loss = self.swap_loss(outputs, targets).cpu()
            except ValueError:
                targets_one_hot = self.to_onehot(targets, outputs.shape[1])
                loss = self.swap_loss(outputs, targets_one_hot).cpu()
                loss = loss.view(loss.size(0), -1)
                loss = loss.mean(-1)
            
            predicts = torch.max(cur_replay_output, dim=1)[1]
            r_predicted = (predicts.cpu() == cur_replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]

            r_idxs = cur_replay_idxs[r_predicted]
            r_loss = loss[r_predicted]            
            r_targets = cur_replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = cur_replay_data_ids[r_predicted]

            sorted_r = torch.argsort(r_loss)[:how_much_swap]
            selected_r_loss = r_loss[sorted_r]

            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]


            if len(sorted_r) < how_much_swap:
                w_predicted = (predicts.cpu() != cur_replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                w_idxs = cur_replay_idxs[w_predicted]
                w_loss = loss[w_predicted]

                w_targets = cur_replay_targets[w_predicted]
                    
                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]

                w_how_much_swap = how_much_swap-len(sorted_r)
                sorted_w = torch.argsort(w_loss)[:w_how_much_swap]

                selected_w_loss = w_loss[sorted_w]

                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]
                    
                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids

        
            if i==0:                
                total_selected_idxs = selected_idxs
                total_selected_targets = selected_targets
                if data_ids is not None:
                    total_selected_data_ids = selected_data_ids

            else:
                total_selected_idxs = torch.cat((total_selected_idxs,selected_idxs),dim=-1)
                total_selected_targets = torch.cat((total_selected_targets,selected_targets),dim=-1)
                if data_ids is not None:
                    total_selected_data_ids = torch.cat((total_selected_data_ids,selected_data_ids),dim=-1)

        
        assert len(total_selected_idxs) == total_how_much_swap

        if data_ids is not None:
            return total_selected_idxs, total_selected_targets, total_selected_data_ids
        else:
            return total_selected_idxs, total_selected_targets


    def hybrid_balanced_p(self, idxs, outputs, targets, data_ids=None):
        
        swap_ratio = self.threshold

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        total_how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))


        #get the number of each class inside the batch
        batch_dist = {}
        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()
        
        for cls in replay_targets:
            if cls.item() not in batch_dist:
                batch_dist[cls.item()] = 1
            else:
                batch_dist[cls.item()] += 1

        expected = 0
        for key in batch_dist.keys():
            batch_dist[key] = math.modf( batch_dist[key] * swap_ratio )
            expected += int(batch_dist[key][1])

        shortage = total_how_much_swap - expected

        if shortage > 0:
            get_dec_samples = list(filter(lambda x: x[1][0] != 0, batch_dist.items()))
            selected_dec_samples = random.sample(get_dec_samples, shortage)
            
            for t in selected_dec_samples:
                k, _ = t
                #k = t
                batch_dist[k] = tuple((batch_dist[k][0] ,batch_dist[k][1] + 1))
        
        if replay_output.nelement() == 0:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        #separate batch into class-wise
        for i, (k,v) in enumerate(batch_dist.items()):

            how_much_swap = int(v[1])
            if how_much_swap == 0:
                continue
            
            
            cur_cls_idx = (replay_targets==k).squeeze().nonzero(as_tuple=True)[0]
            
            cur_replay_output = replay_output[cur_cls_idx]
            cur_replay_idxs = replay_idxs[cur_cls_idx]
            cur_replay_targets = replay_targets[cur_cls_idx]            
            if data_ids is not None:
                cur_replay_data_ids = replay_data_ids[cur_cls_idx]
            
            soft_output = self.softmax(cur_replay_output)
            entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
            
            predicts = torch.max(cur_replay_output, dim=1)[1]
            r_predicted = (predicts.cpu() == cur_replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]

            r_idxs = cur_replay_idxs[r_predicted]
            r_entropy = entropy[r_predicted]
            r_targets = cur_replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = cur_replay_data_ids[r_predicted]

            sorted_r = torch.argsort(r_entropy)[:how_much_swap]


            selected_r_entropy = r_entropy[sorted_r]


            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]


            if len(sorted_r) < how_much_swap:

                w_predicted = (predicts.cpu() != cur_replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                w_idxs = cur_replay_idxs[w_predicted]
                w_entropy = entropy[w_predicted]
                w_targets = cur_replay_targets[w_predicted]
                    
                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]

                
                w_how_much_swap = how_much_swap-len(sorted_r)
                sorted_w = torch.argsort(w_entropy, descending=True)[:w_how_much_swap]
                selected_w_entropy = w_entropy[sorted_w]

                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]
                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids

            try:
                total_selected_idxs = torch.cat((total_selected_idxs,selected_idxs),dim=-1)
                total_selected_targets = torch.cat((total_selected_targets,selected_targets),dim=-1)
                if data_ids is not None:
                    total_selected_data_ids = torch.cat((total_selected_data_ids,selected_data_ids),dim=-1)

            except:                
                total_selected_idxs = selected_idxs
                total_selected_targets = selected_targets
                if data_ids is not None:
                    total_selected_data_ids = selected_data_ids

        assert len(total_selected_idxs) == total_how_much_swap

        if data_ids is not None:
            return total_selected_idxs, total_selected_targets, total_selected_data_ids
        else:
            return total_selected_idxs, total_selected_targets



    def hybrid_balanced(self, idxs, outputs, targets, data_ids=None):
        
        swap_ratio = self.threshold

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))

        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        soft_output = self.softmax(replay_output)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        if replay_output.nelement() != 0:
            predicts = torch.max(replay_output, dim=1)[1]
            r_predicted = (predicts.cpu() == replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]

            r_idxs = replay_idxs[r_predicted]
            r_entropy = entropy[r_predicted]
            r_targets = replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = replay_data_ids[r_predicted]

            sorted_r_org = torch.argsort(r_entropy)

            selected = []
            filled_counter = 0

            for i, idx in enumerate(sorted_r_org):

                if filled_counter >= how_much_swap:
                    break

                label = r_targets[idx].item()
                if label in self.swap_class_dist:
                    if self.swap_class_dist[label] + 1 <= self.swap_thr:
                        self.swap_class_dist[label] += 1
                        filled_counter +=1
                        selected.append(i)
                    else:
                        continue
                else:
                    self.swap_class_dist[label] = 1
                    filled_counter +=1
                    selected.append(i)

            sorted_r = sorted_r_org[selected][:how_much_swap]

            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]

            if len(sorted_r) < how_much_swap:

                w_predicted = (predicts.cpu() != replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                w_idxs = replay_idxs[w_predicted]
                w_entropy = entropy[w_predicted]
                w_targets = replay_targets[w_predicted]
                    
                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]

                w_how_much_swap = how_much_swap-len(sorted_r)
                sorted_w_org = torch.argsort(w_entropy, descending=True)


                selected = []
                filled_counter = 0

                for i, idx in enumerate(sorted_w_org):
                    if filled_counter >= w_how_much_swap:
                        break
                    label = w_targets[idx].item()
                    if label in self.swap_class_dist:
                        if self.swap_class_dist[label] + 1 <= self.swap_thr:
                            self.swap_class_dist[label] += 1
                            filled_counter +=1
                            selected.append(i)
                        else:
                            continue
                    else:
                        self.swap_class_dist[label] = 1
                        filled_counter +=1
                        selected.append(i)
                    
                sorted_w = sorted_w_org[selected][:w_how_much_swap]
                    
                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]

                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids
        
        else:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        if len(selected_idxs) != how_much_swap:
            for unselected in sorted_r_org:
                
                if len(selected_idxs) == how_much_swap:
                    break

                if unselected not in sorted_r:
                    selected_idxs = torch.cat((selected_idxs,r_idxs[unselected].reshape(1)),dim=-1)
                    selected_targets = torch.cat((selected_targets,r_targets[unselected].reshape(1)),dim=-1)
                    if data_ids is not None:
                        selected_data_ids = torch.cat((selected_data_ids,r_data_ids[unselected].reshape(1)),dim=-1)
                    self.swap_class_dist[r_targets[unselected].item()] += 1
            
            for unselected in sorted_w_org:
                if len(selected_idxs) == how_much_swap:
                    break

                if unselected not in sorted_w:
                    selected_idxs = torch.cat((selected_idxs,w_idxs[unselected].reshape(1)),dim=-1)
                    selected_targets = torch.cat((selected_targets,w_targets[unselected].reshape(1)),dim=-1)
                    if data_ids is not None:
                        selected_data_ids = torch.cat((selected_data_ids,w_data_ids[unselected].reshape(1)),dim=-1)
                    self.swap_class_dist[w_targets[unselected].item()] += 1

        assert len(selected_idxs) == how_much_swap

        if data_ids is not None:
            return selected_idxs, selected_targets, selected_data_ids
        else:
            return selected_idxs, selected_targets

    
    def all(self, idxs, outputs=None, targets=None, data_ids=None):
    
        if targets is not None and data_ids is not None:   
            return self.get_replay_index(idxs, targets, data_ids)
        else:
            return self.get_replay_index(idxs, targets)
            

    def hybrid_random(self, idxs, outputs, targets, data_ids=None):
        
        swap_ratio = self.threshold
    
        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))

        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        soft_output = self.softmax(replay_output)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        
        if replay_output.nelement() != 0:
            predicts = torch.max(replay_output, dim=1)[1]
            r_predicted = (predicts.cpu() == replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]

            r_idxs = replay_idxs[r_predicted]
            r_entropy = entropy[r_predicted]
            r_targets = replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = replay_data_ids[r_predicted]

            sorted_r = torch.argsort(r_entropy)[:how_much_swap]

            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]

            if len(sorted_r) < how_much_swap:

                w_predicted = (predicts.cpu() != replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                w_idxs = replay_idxs[w_predicted]
                w_entropy = entropy[w_predicted]
                w_targets = replay_targets[w_predicted]
                    
                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]

                w_how_much_swap = how_much_swap-len(sorted_r)

                idx_to_pick = np.linspace(0, len(w_idxs), num=w_how_much_swap, dtype = int, endpoint=False)
                sorted_w = torch.argsort(w_entropy, descending=True)[idx_to_pick]

                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]

                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids
        
        else:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        assert len(selected_idxs) == how_much_swap

        if data_ids is not None:
            return selected_idxs, selected_targets, selected_data_ids
        else:
            return selected_idxs, selected_targets
def fetch_from_storage_imagenet(replay_dataset, swap_idxs,labels,task_id,filenames=None,testset='imagenet1000',transform=None):
    skipped = 0
    for i in range(len(swap_idxs)):
        idx = swap_idxs[i]
        label = labels[i]
        path_curr_label = replay_dataset.rb_path + '/' + str(label)
        try:
            prefix = f'{label}_'
            suffix = '.png'
            num_file = IMG1000_INDEX[label]
            if filenames ==None:
                replace_file = path_curr_label + '/' + prefix+str(random.randint(1,num_file)) + '.png'
            else: 
                file_no = eval(filenames[i].split('_')[-1])
                if file_no > num_file:replace_file = path_curr_label + '/' + prefix+str(random.randint(1,num_file)) + '.png'
                else: replace_file = path_curr_label + '/' +filenames[i] + '.png'
        except Exception as e:
            raise
        if idx < len(replay_dataset) and replace_file == replay_dataset.filenames[idx]:
            skipped+=1
        else:
            get_file_imagenet(idx, replace_file,replay_dataset,transform=transform)
def get_file_imagenet(idx, filename, replay_dataset,transform=None):
    if 'png' in filename:
        vec = _get_img_imagenet(filename)
    else:
        vec = np.load(filename)
        
    swap_in_file = filename.split('/')[-1]
    swap_in_file = swap_in_file.split('.png')[0]
    if idx < len(replay_dataset.data):
        replay_dataset.data[idx] = vec
        replay_dataset.filenames[idx] = swap_in_file
    else: 
        replay_dataset.data.append(vec)
        replay_dataset.filenames.append(swap_in_file)
    return True
def _get_img_imagenet(filename):
    f = os.open( filename, os.O_RDONLY | os.O_DIRECT)

    os.lseek(f,0,0)
    actual_size = os.path.getsize(filename)
    block_size = 512 * math.ceil(actual_size / 512)
    fr = directio.read(f, block_size)
    os.close(f)
    
    data = io.BytesIO(fr[:actual_size])
    

    img = Image.open(data)
    img = img.convert('RGB')
    
    return img