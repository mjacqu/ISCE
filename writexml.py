def makexml(orbits,auxfiles,masterfile,slavefile,swaths,az,rng):
    from lxml import etree
    import os
    root = etree.Element('topsApp')
    child = etree.SubElement(root, "component", name="topsinsar")
    subchild1 = etree.SubElement(child, "component", name="master")
    subsubchild1 = etree.SubElement(subchild1, "property", name="orbit directory")
    subsubchild1.text = str(orbits)
    subsubchild2 = etree.SubElement(subchild1, "property", name="auxiliary data directory")
    subsubchild2.text = str(auxfiles)
    subsubchild3 = etree.SubElement(subchild1, "property", name="safe")
    subsubchild3.text = str(masterfile) #substitute with funct. parameter
    subsubchild4 = etree.SubElement(subchild1, "property", name="output directory")
    subsubchild4.text = "master"
    subchild2 = etree.SubElement(child, "component", name="slave")
    subsubchild5 = etree.SubElement(subchild2, "property", name="orbit directory")
    subsubchild5.text = str(orbits)
    subsubchild6 = etree.SubElement(subchild2, "property", name="auxiliary data directory")
    subsubchild6.text = str(auxfiles)
    subsubchild7 = etree.SubElement(subchild2, "property", name="safe")
    subsubchild7.text = str(slavefile) #substitute with funct. parameter
    subsubchild8 = etree.SubElement(subchild2, "property", name="output directory")
    subsubchild8.text = "slave"
    subchild3 = etree.SubElement(child, "property", name="swaths")
    subchild3.text = str(swaths)
    subchild4 = etree.SubElement(child, "property", name="azimuth looks")
    subchild4.text = str(az) #substitute with funct. parameter
    subchild5 = etree.SubElement(child, "property", name="range looks")
    subchild5.text = str(rng)
    subchild6 = etree.SubElement(child, "property", name="Sensor name")
    subchild6.text = "SENTINEL1"
    subchild7 = etree.SubElement(child, "property", name="unwrapper name")
    subchild7.text = "snaphu_mcf"
    subchild8 = etree.SubElement(child, "property", name="do unwrap")
    subchild8.text = "True"
    # subchild9 = etree.SubElement(child, "property", name="demfilename")
    # subchild9.text = "/mnt/MyleneShare/Chile/data/asc/30mdem/demLat_S45_S42_Lon_W074_W071.dem.wgs84"
    # subchild10 = etree.SubElement(child, "property", name="region of interest")
    # subchild10.text = "[-43.432981,-43.3236,-72.52682,-72.32925]"
    # subchild11 = etree.SubElement(child, "property", name="geocode bounding box")
    # subchild11.text = "[-43.432981,-43.3236,-72.52682,-72.32925]"
    # make string pretty
    s = etree.tostring(root, pretty_print=True) # ensure pretty formatting
    xml = open("topsApp.xml", "w") # now create basic topsApp.xml file in the directory
    xml.write(s) # write contents of s into xml file
    xml.close() # close xml file in order to save
