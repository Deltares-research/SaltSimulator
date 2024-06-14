import pandas as pd
import numpy as np

from collections import namedtuple, defaultdict

from ertoolbox.lib import my_timer
from ertoolbox.lib import db_connect

def read_dpid_mapper(database, ids):
    with db_connect(database) as connection:
        cursor = connection.cursor()
        
        cursor.execute("SELECT Value FROM TaskSettings \
                       WHERE Setting='ElectrodeSpacing' AND key1=1")
        result = cursor.fetchall()
        for row in result:
            spacing = list(map(float, row[0].split(';')))
            break  # Single line querry anyway
        #print(spacing)
        
        cursor.execute("SELECT ID, APosX, BPosX, MPosX, NPosX \
                        FROM DP_ABMN WHERE TaskID in {}".format(str(ids)))
        result = cursor.fetchall()
        dpid_abmn_lookup = dict()        
        for row in result:
            dpid = row[0]
            abmn = row[1:5]
            dpid_abmn_lookup[dpid] = [elec*spacing[0] for elec in abmn]
        return dpid_abmn_lookup

def read_geometry_mapper(database, ids):
    with db_connect(database) as connection:
        cursor = connection.cursor()
        
        cursor.execute("SELECT DPID FROM DPV \
                       WHERE DatatypeID=5 AND Channel>0 \
                       AND TaskID in {}".format(str(ids)))
        result = cursor.fetchall()
        
        geometry_lookuptable = dict()
        geometry_lookuptable_reverse = dict()
        for index, row in enumerate(result):
            key = row[0]
            geometry_lookuptable[key] = index
            geometry_lookuptable_reverse[index] = key
        return geometry_lookuptable, geometry_lookuptable_reverse

def read_focus_point_mapper(database, ids):
    with db_connect(database) as connection:
        cursor = connection.cursor()
        
        cursor.execute("SELECT ID, FocusX, FocusZ \
                        FROM DP_ABMN WHERE TaskID in {}".format(str(ids)))
        result = cursor.fetchall()
        focus_point_lookup = dict()        
        for row in result:
            dpid = row[0]
            focus_point_lookup[dpid] = (row[1], row[2])
        return focus_point_lookup

def read_task_mapper(database, ids):

    task_dpid_lookup = defaultdict(list)
    task_dpid_lookup_reverse = dict()

    with db_connect(database) as connection:
        cursor = connection.cursor()
        
        cursor.execute("SELECT TaskID, ID \
                        FROM DP_ABMN WHERE TaskID in {} \
                        ORDER BY ID".format(str(ids)))
        result = cursor.fetchall()
        
        for row in result:
            
            task = row[0]
            dpid = row[1]
            task_dpid_lookup[task].append(dpid)
            task_dpid_lookup_reverse[dpid] = task
        
        return task_dpid_lookup, task_dpid_lookup_reverse
    
def read_task(db='project.db', ids=None, includeIP=False):
    if (ids is None) or (not isinstance(ids, tuple)) :
        raise ValueError('Task ID needs to be a tuple')
    if len(ids) == 0:
        raise ValueError('Provide at least one task id')
    with db_connect(db) as connection:
        cursor = connection.cursor()
        # read info
        tinfo = task_info(cursor, ids[0])
        if tinfo == -1:
            return None
        # read meas
        # TODO: data should be appended to task info
        data = meas_info(cursor, tinfo, ids)
        data['charg'] = integral(data, tinfo.gates, sgate=1, egate=0)
        return data
    
def task_info(cursor, id):
    cursor.execute('''SELECT DISTINCT ID, Name, SpacingX, SpacingY, SpacingZ, ArrayCode, nmeas,
                                (CASE WHEN Setting LIKE 'IP_WindowSecList' THEN AcqSettings.Value ELSE 0 END) AS gates
                            FROM Tasks
                            JOIN AcqSettings
                            ON Tasks.ID = AcqSettings.key1
                            JOIN
                                (SELECT TaskID, COUNT(*) AS nmeas
                                FROM
                                    (SELECT TaskID FROM DPV WHERE DatatypeID=3 GROUP BY MeasureID, Channel) AS t
                                GROUP BY TaskID) AS TT
                            ON TT.TaskID = Tasks.ID
                            WHERE gates <> 0 AND ID=?
                            ''', (id,))
    TaskTuple = namedtuple('Task', 'id Name SpacingX SpacingY SpacingZ ArrayCode nmeas gates')
    temp = cursor.fetchall()
    if len(temp) == 0:
        print('task not found')
        return -1
    result = list(temp[0])
    result[-1] = np.array([float(gate) for gate in result[-1].split()])  # gates from str to list of floats
    tinfo = TaskTuple._make(result)
    return tinfo


def meas_info(cursor, tinfo, ids):
    n = len(tinfo.gates) - 1
    ipquery = ''
    for i in range(1, n + 1):
        ipquery += ",sum(CASE WHEN SeqNum = {0:d} AND DatatypeID=3 THEN DPV.DataValue ELSE 0 END) AS IP{0:d}".format(i)
    ipquery += ",DataSDev"
    for i in range(1, n + 1):
        ipquery += ",sum(CASE WHEN SeqNum = {0:d} AND DatatypeID=3 THEN DataSDev ELSE 0 END) AS SD{0:d}".format(i)
    cursor.execute("\
        SELECT Time, DPV.TaskID, DPV.MeasureID, DPV.DPID, \
           APosX, APosY, APosZ, BPosX, BPosY, BPosZ, \
           MPosX, MPosY, MPosZ, NPosX, NPosY, NPosZ, \
           FocusX, FocusY, FocusZ, Channel, \
        sum(CASE WHEN DatatypeID = 7 THEN DPV.Datavalue ELSE 0 END) AS voltage, \
        injections.DataValue as current, \
        sum(CASE WHEN DatatypeID = 5 THEN DPV.Datavalue ELSE 0 END) AS res, \
        sum(CASE WHEN DatatypeID = 2 THEN DPV.Datavalue ELSE 0 END) AS apres \
        {} \
        FROM DPV \
        INNER JOIN Datatype \
        ON DPV.DatatypeID = Datatype.ID \
        INNER JOIN Measures \
        ON DPV.MeasureID = Measures.ID \
        INNER JOIN DP_ABMN \
        ON DPV.DPID = DP_ABMN.ID \
        INNER JOIN (SELECT DPV.MeasureID, DPV.DataValue \
                    FROM DPV \
                    WHERE DPV.DatatypeID=6 AND DPV.Channel=14) injections\
        ON DPV.MeasureID = injections.MeasureID \
        WHERE Channel NOT IN (0, 13, 14) AND DPV.TaskID in {}\
        GROUP BY DPV.MeasureID, Channel \
        ORDER BY DPID --MeasureID, Channel \n\
        --LIMIT 10;".format(ipquery, str(ids)))
    # Save data in pandas DataFrame object
    str_label = "Time TaskID MeasureID DPID APosX APosY APosZ BPosX BPosY BPosZ MPosX MPosY MPosZ NPosX NPosY NPosZ FocusX FocusY FocusZ Channel \
                 volt current res apres"
    for i in range(1, n+1):
        str_label += " IP{}".format(i)
    str_label += " SDev"
    for i in range(1, n+1):
        str_label += " SD{}".format(i)
    labels = [label for label in str_label.split()]
    data = pd.DataFrame(cursor.fetchall(), columns=labels)
    data['Time'] = pd.to_datetime(data['Time'])
    data[['APosX', 'BPosX', 'MPosX', 'NPosX']] *= tinfo.SpacingX
    data[['APosY', 'BPosY', 'MPosY', 'NPosY']] *= tinfo.SpacingY
    data[['APosZ', 'BPosZ', 'MPosZ', 'NPosZ']] *= tinfo.SpacingZ
    return data


def integral(data, gates_width, sgate=1, egate=0):
    sgatedata = data.columns.get_loc('IP1')
    egatedata = data.columns.get_loc('SDev')
    if egate == 0:
        egate = egatedata - sgatedata
    if sgate > egate:
        raise ValueError('Starting gate needs to be smaller than end gate')
    sgate -= 1  # 'Zero-Index' data
    ngates = egate - sgate
    widths = gates_width[1+sgate:1+egate]
    charg = (data.iloc[:, sgatedata+sgate:sgatedata+sgate+ngates] * widths).apply(sum, axis=1) / sum(widths) * 1000
    return charg


def find_gates(data, sgate=1, egate=0):
    sgatedata = data.columns.get_loc('IP1')
    egatedata = data.columns.get_loc('SDev')
    if egate == 0:
        egate = egatedata - sgatedata
    if sgate > egate:
        raise ValueError('Starting gate needs to be smaller than end gate')
    sgate -= 1  # 'Zero-Index' data
    ngates = egate - sgate
    sgateloc = sgatedata + sgate
    egateloc = sgatedata + sgate + ngates
    return sgateloc, egateloc


def find_gates_error(data, sgate=1, egate=0):
    sgatedata = data.columns.get_loc('SD1')
    egatedata = len(data.columns)
    if egate == 0:
        egate = egatedata - sgatedata
    if sgate > egate:
        raise ValueError('Starting gate needs to be smaller than end gate')
    sgate -= 1  # 'Zero-Index' data
    ngates = egate - sgate
    sgateloc = sgatedata + sgate
    egateloc = sgatedata + sgate + ngates
    return sgateloc, egateloc