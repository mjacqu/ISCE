import glob, os
from lxml import etree
import re
import itertools
import datetime
import glob
import numpy as np

class Pair(object):
    """
    Attributes:
        reference (str): Path to reference data file
        secondary (str): Path to secondary data file
        swaths (list): list of swaths to process, i.e. [2,3]
        orbit (str): Path to orbit directory
        auxiliary (str): Path to auxiliary directory
        path (str): Where to put pairs directory
        unwrapper (str): snaphu_mcf (default), grass, icu, snaphu
        unwrap (str): Default is True
        az_looks (int): Number of looks in azimuth. Default is 7
        rng_looks (int): Number of looks in range. Default is 19.
        dem (str): Path to dem file. Default is None (i.e. automatic download)
        roi (list): list of region of interest coordinates [S,N,W,E]
        bbox (list): list of bounding box coordinates [S,N,W,E]
        sensor (str): default is 'SENTINEL1'
        dense_offsets (str): Default is False
        polarization (str): Default is vv
    """
    def __init__(self, reference, secondary, path, orbit, auxiliary, polarization='vv', swaths=None,
        unwrapper='snaphu_mcf', unwrap=True, az_looks=None, rng_looks=None,
        dem=None, roi=None, bbox=None, sensor='SENTINEL1', dense_offsets=False):
        self.path = os.path.join(path, safe2date(reference).strftime('%Y%m%d') + '_' + safe2date(secondary).strftime('%Y%m%d'))
        self.reference = reference
        self.secondary = secondary
        self.swaths = swaths
        self.unwrapper = unwrapper
        self.orbit = orbit
        self.auxiliary = auxiliary
        self.polarization = polarization
        self.unwrap = unwrap
        self.az_looks = az_looks
        self.rng_looks = rng_looks
        self.dem = dem
        self.roi = roi
        self.bbox = bbox
        self.sensor = sensor
        self.dense_offsets = dense_offsets


    @classmethod
    def from_path(cls, path):
        xml_fn = 'topsApp.xml'
        root = etree.parse(os.path.join(path,xml_fn)).getroot()
        return cls(
            reference = root.xpath('component[@name="topsinsar"]/component[@name="reference"]/property[@name="safe"]')[0].text,
            secondary = root.xpath('component[@name="topsinsar"]/component[@name="secondary"]/property[@name="safe"]')[0].text,
            swaths = root.xpath('component[@name="topsinsar"]/property[@name="swaths"]')[0].text,
            orbit = root.xpath('component[@name="topsinsar"]/component[@name="reference"]/property[@name="orbit directory"]')[0].text,
            auxiliary = root.xpath('component[@name="topsinsar"]/component[@name="reference"]/property[@name="auxiliary data directory"]')[0].text,
            path = os.path.dirname(path),
            #polarization = root.xpath('component[@name="topsinsar"]/component[@name="reference"]/property[@name="polarization"]')[0].text,
            #polarization = root.xpath('component[@name="topsinsar"]/component[@name="secondary"]/property[@name="polarization"]')[0].text,
            unwrapper = root.xpath('component[@name="topsinsar"]/property[@name="unwrapper name"]')[0].text,
            unwrap = root.xpath('component[@name="topsinsar"]/property[@name="do unwrap"]')[0].text,
            az_looks = root.xpath('component[@name="topsinsar"]/property[@name="azimuth looks"]')[0].text,
            rng_looks = root.xpath('component[@name="topsinsar"]/property[@name="range looks"]')[0].text,
            dem = root.xpath('component[@name="topsinsar"]/property[@name="demfilename"]')[0].text,
            roi = root.xpath('component[@name="topsinsar"]/property[@name="region of interest"]')[0].text,
            bbox = root.xpath('component[@name="topsinsar"]/property[@name="geocode bounding box"]')[0].text,
            sensor = root.xpath('component[@name="topsinsar"]/property[@name="Sensor name"]')[0].text,

            #dense_offsets = root.dense_offsets('component[@name="topsinsar"]/property[@name="do denseOffsets"]')[0].text
        )

    def as_xml(self):
        reference = dict(
            name='reference',
            property=[
                dict(name='orbit directory', __text__=self.orbit),
                dict(name='auxiliary data directory', __text__=self.auxiliary),
                dict(name='safe', __text__=self.reference),
                dict(name='polarization', __text__=self.polarization),
                dict(name='output directory', __text__='reference')]
            )
        secondary = dict(
            name='secondary',
            property=[
                dict(name='orbit directory', __text__=self.orbit),
                dict(name='auxiliary data directory', __text__=self.auxiliary),
                dict(name='safe', __text__=self.secondary),
                dict(name='polarization', __text__=self.polarization),
                dict(name='output directory', __text__='secondary')]
            )
        properties = [
            dict(name='Sensor name', __text__=self.sensor),
            dict(name='do unwrap', __text__=str(self.unwrap)),
            dict(name='swaths', __text__=str(self.swaths)),
            dict(name='unwrapper name', __text__=str(self.unwrapper))
        ]
        if self.az_looks:
            properties.append(dict(
                name='azimuth looks', __text__=str(self.az_looks)))
        if self.rng_looks:
            properties.append(dict(
                name='range looks', __text__=str(self.rng_looks)))
        if self.dem:
            properties.append(dict(
                name='demfilename', __text__=self.dem))
        if self.roi:
            properties.append(dict(
                name='region of interest', __text__=str(self.roi)))
        if self.bbox:
            properties.append(dict(
                name='geocode bounding box', __text__=str(self.bbox)))
        if self.dense_offsets:
            properties.append(dict(
                name='do denseOffsets', __text__=str(self.dense_offsets)))
        tops_dict = dict(
            topsApp=dict(
                component=dict(name='topsinsar',
                    component=[reference, secondary],
                    property=properties
                )
            )
        )
        tops_xml = d2xml(tops_dict)
        return etree.tostring(tops_xml, pretty_print=True, encoding=str) # ensure pretty formatting

    def make_path(self):
        os.makedirs(self.path, exist_ok=True)

    def write_xml(self, overwrite=False):
        xml_path = os.path.join(self.path, 'topsApp.xml')
        if not overwrite and os.path.isfile(xml_path):
            raise ValueError('File already exists: ' + xml_path + ' Set overwrite=True to overwrite.')
        xml = self.as_xml()
        with open(xml_path, 'w') as fp:
            fp.write(xml)

    def run(self, overwrite=False):
        self.make_path()
        self.write_xml(overwrite=overwrite)
        original_dir = os.getcwd()
        os.chdir(self.path)
        os.system('topsApp.py --steps')
        os.chdir(original_dir)

    def check_process(self):
        check = os.path.exists(os.path.join(self.path,'merged'))
        return check
        #if check == True:
        #    print(os.path.basename(self.path) + ' processed')
        #else:
        #    print(os.path.basename(self.path) + ' failed')

    def ls(self):
        pass


# ---- Helpers ----


def safe2date(path):
    """
    Parse datetime from each safe-file path.
    """
    m = re.search(r'[A-Z]{3}_([0-9]{8}T[0-9]{6})_([0-9]{8}T[0-9]{6})_', path)
    if len(m.groups()) == 2:
        start = datetime.datetime.strptime(m.group(1), '%Y%m%dT%H%M%S')
        end = datetime.datetime.strptime(m.group(2), '%Y%m%dT%H%M%S')
        return start + (end - start) * 0.5
    else:
        raise ValueError('Failed to parse datetime from path: ' + path)

# Create python xml structures compatible with
# http://search.cpan.org/~grantm/XML-Simple-2.18/lib/XML/Simple.pm
def d2xml(d):
    """
    convert dict to xml

    1. The top level d must contain a single entry i.e. the root element
    2.  Keys of the dictionary become sublements or attributes
    3.  If a value is a simple string, then the key is an attribute
    4.  if a value is dict then, then key is a subelement
    5.  if a value is list, then key is a set of sublements

    a  = { 'module' : {'tag' : [ { 'name': 'a', 'value': 'b'},
                                { 'name': 'c', 'value': 'd'},
                             ],
                      'gobject' : { 'name': 'g', 'type':'xx' },
                      'uri' : 'test',
                   }
       }
    >>> d2xml(a)
    <module uri="test">
       <gobject type="xx" name="g"/>
       <tag name="a" value="b"/>
       <tag name="c" value="d"/>
    </module>

    @type  d: dict
    @param d: A dictionary formatted as an XML document
    @return:  A etree Root element
    """
    def _d2xml(d, p):
        for k,v in d.items():
            if isinstance(v,dict):
                node = etree.SubElement(p, k)
                _d2xml(v, node)
            elif isinstance(v,list):
                for item in v:
                    node = etree.SubElement(p, k)
                    _d2xml(item, node)
            elif k == "__text__":
                    p.text = v
            elif k == "__tail__":
                    p.tail = v
            else:
                p.set(k, v)

    key = list(d.keys())[0]
    root = etree.Element(key)
    _d2xml(d[key], root)
    return root

# def csv_to_datelist(file_name):
#     list_of_dates = pd.read_csv(file_name, delimiter = ',', header = None).to_numpy()
#     #timestamps = pd.to_datetime(dates.stack(), format = '%Y%m%d').unstack()
#     #list_of_dates = [tuple(row) for row in dates.values]
#     return list_of_dates

def make_pairs(path, maxdelta=None, singleref=None, dates=None, sequential = None, options=dict()):
    """
    Returns an array of Pair objects to pass to run().

    Arguments:
    path (str):                                  path to raw data directory with zipped safe files
    maxdelta (datetime.timedelta):               Maximum time delta between paired acquisition
                                                 dates.
    singleref (datetime.datetime):               date to use as single reference image
    dates (list of datetime.datetime tuples):    pass specific pairs of dates to process
    sequential (int):                            nth date to pair with each date
    options (dict):                              see Pair() class definition for details

    Returns:
    Pair (object)
    """
    def make_pair(first, second):
        first, second = np.atleast_1d(first), np.atleast_1d(second)
        pairs = []
        for i, j in zip(first, second):
            if datetimes[i] > datetimes[j]:
                i, j = j, i
            pairs.append((paths[i], paths[j]))
        pairs = list(set([p for p in pairs]))
        return pairs

    paths = glob.glob(os.path.join(path, "S1*.zip"))
    datetimes = np.asarray([safe2date(i) for i in paths])


    ref = []
    sub = []
    # make pairs from dates
    if dates is not None:
        datetimes_str = [d.strftime('%Y%m%d') for d in datetimes]
        pairs = []
        for first, second in dates:
            r, s = [datetimes_str.index(first)], [datetimes_str.index(second)]
            ref.extend(r)
            sub.extend(s)
    #make pairs with singleref
    if maxdelta is not None:
        matches = np.abs(np.column_stack([datetimes - d for d in datetimes])) <= abs(maxdelta+datetime.timedelta(days = 1))
        #first, second = np.where(np.triu(matches, k=1))
        r, s = list(np.where(np.triu(matches, k=1)))
        ref.extend(r)
        sub.extend(s)
    #make pairs sequentially (each date with every nth image)
    if sequential is not None:
        sorted_dates = sorted(datetimes)
        seq_ref = []
        seq_sub = []
        for d in range(0, len(sorted_dates)-sequential):
            r, s = list(np.where(datetimes == sorted_dates[d])[0]), list(np.where(datetimes == sorted_dates[d+sequential])[0]) # need to add option for every third etc.
            seq_ref.extend(r[:])
            seq_sub.extend(s[:])
        ref.extend(seq_ref)
        sub.extend(seq_sub)
    # make pairs with one single reference image
    if singleref is not None:
        filter = [d.date() == singleref.date() for d in datetimes]
        ref_date = list(itertools.compress(datetimes, filter))
        sm_reference = np.where(datetimes == ref_date)[0]
        if len(sm_reference) == 0:
            raise Exception ('No reference file by this date found: ' + str(singleref.date()))
        if len(sm_reference) > 1:
            raise Exception ('More than one reference file found. Limit to one file.')
        secondary = list(np.setdiff1d(datetimes, ref_date, assume_unique=True))
        temp = set(secondary)
        sm_sub = [i for i, val in enumerate(datetimes) if val in temp]
        sm_ref = list(sm_reference) * len(secondary)
        ref.extend(sm_ref)
        sub.extend(sm_sub)
    all_pairs = make_pair(ref, sub)
    return [Pair(reference=m, secondary=s, **options) for m, s in all_pairs]
    # make pairs with maxdelta
    #if maxdelta is None:
    #    matches = np.abs(np.column_stack([datetimes - d for d in datetimes])) >= abs(datetime.timedelta(days=0))
    #    first, second = np.where(np.triu(matches, k=1))
    #    pairs = make_pair(first, second)
    #    return [Pair(reference=m, secondary=s, **options) for m, s in pairs]
