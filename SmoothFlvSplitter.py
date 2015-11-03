#!/usr/bin/env python

##
##  flv.py - reading/writing FLV file format.
##
##  Copyright (c) 2009-2010 by Yusuke Shinyama
##

import sys
import os
import numpy as np
from struct import pack, unpack
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO


# return the number of required bits for x.
def needbits1(x, signed=False):
    if x == 0:
        return 0
    if signed:
        n = 1
        if x < 0:
            x = -x - 1
    else:
        n = 0
        assert 0 < x
    while True:
        n += 1
        x >>= 1
        if x == 0:
            break
    return n


def needbits(args, signed=False):
    return max([needbits1(x, signed) for x in args])
# assert needbits1(0,0) == 0
# assert needbits1(0,1) == 0
# assert needbits1(1,0) == 1
# assert needbits1(1,1) == 2
# assert needbits1(2,0) == 2
# assert needbits1(-2,1) == 2
# assert needbits1(-3,1) == 3
# assert needbits1(127,0) == 7
# assert needbits1(127,1) == 8
# assert needbits1(128,0) == 8
# assert needbits1(-128,1) == 8
# assert needbits1(-129,1) == 9
# assert needbits1(-6380,1) == 14

# get value


def getvalue(fp):
    t = fp.read(1)
    if not t:
        raise EOFError
    elif t == '\x00':
        (n,) = unpack('>d', fp.read(8))
        return n
    elif t == '\x01':
        return bool(ord(fp.read(1)))
    elif t == '\x02' or t == '\x04':
        (n,) = unpack('>H', fp.read(2))
        return fp.read(n)
    elif t == '\x03':
        d = {}
        try:
            while True:
                (n,) = unpack('>H', fp.read(2))
                if n == 0:
                    assert fp.read(1) == '\x09'
                    break
                k = fp.read(n)
                v = getvalue(fp)
                d[k] = v
        except (error, EOFError):
            pass
        return d
    elif t == '\x07':
        (n,) = unpack('>H', fp.read(2))
        return n
    elif t == '\x08':
        (n,) = unpack('>L', fp.read(4))
        d = {}
        for _ in xrange(n):
            (n,) = unpack('>H', fp.read(2))
            k = fp.read(n)
            v = getvalue(fp)
            d[k] = v
        return d
    elif t == '\x0a':
        (n,) = unpack('>L', fp.read(4))
        return [getvalue(fp) for _ in xrange(n)]
    elif t == '\x0b':
        return fp.read(10)
    elif t == '\x0c':
        (n,) = unpack('>L', fp.read(4))
        return fp.read(n)
    else:
        return None


# DataParser
##
class DataParser(object):

    def __init__(self, fp, debug=0):
        self.fp = fp
        self.buff = 0
        self.bpos = 8
        self.debug = debug
        return

    def close(self):
        return

    # fixed bytes read

    def read(self, n):
        x = self.fp.read(n)
        if len(x) != n:
            raise EOFError
        return x

    def readui8(self):
        return ord(self.read(1))

    def readsi8(self):
        return unpack('<b', self.read(1))[0]

    def readui16(self):
        return unpack('<H', self.read(2))[0]

    def readub16(self):
        return unpack('>H', self.read(2))[0]

    def readsi16(self):
        return unpack('<h', self.read(2))[0]

    def readub24(self):
        return unpack('>L', '\x00' + self.read(3))[0]

    def readui32(self):
        return unpack('<L', self.read(4))[0]

    def readub32(self):
        return unpack('>L', self.read(4))[0]

    def readrgb(self):
        return (self.readui8(), self.readui8(), self.readui8())

    def readrgba(self):
        return (self.readui8(), self.readui8(), self.readui8(), self.readui8())

    # fixed bits read

    def setbuff(self, bpos=8, buff=0):
        (self.bpos, self.buff) = (bpos, buff)
        return

    def readbits(self, bits, signed=False):
        if bits == 0:
            return 0
        bits0 = bits
        v = 0
        while True:
            # the number of the remaining bits we can get from the current
            # byte.
            r = 8 - self.bpos
            if bits <= r:
                # |-----8-bits-----|
                # |-bpos-|-bits-|  |
                # |      |----r----|
                v = (
                    v << bits) | (
                    (self.buff >> (
                        r -
                        bits)) & (
                        (1 << bits) -
                        1))
                self.bpos += bits
                break
            else:
                # |-----8-bits-----|
                # |-bpos-|---bits----...
                # |      |----r----|
                v = (v << r) | (self.buff & ((1 << r) - 1))
                bits -= r
                self.buff = ord(self.read(1))
                self.bpos = 0
        if signed and (v >> (bits0 - 1)):
            v -= (1 << bits0)
        return v

    # variable length structure

    def readstring(self):
        s = []
        while True:
            c = self.read(1)
            if c == '\x00':
                break
            s.append(c)
        return unicode(''.join(s), self.encoding)


# DataWriter
# A common superclass for SWFWriter and FLVWriter
##
class DataWriter(object):

    def __init__(self, fp, debug=0):
        self.fp = fp
        self.bpos = 0
        self.buff = 0
        self.fpstack = []
        self.debug = debug
        return

    def push(self):
        self.fpstack.append(self.fp)
        self.fp = StringIO()
        return

    def pop(self):
        assert self.fpstack, 'empty fpstack'
        self.fp.seek(0)
        data = self.fp.read()
        self.fp = self.fpstack.pop()
        return data

    def close(self):
        self.finishbits()
        assert not self.fpstack, 'fpstack not empty'
        return

    # fixed bytes write

    def write(self, *args):
        for x in args:
            self.fp.write(x)
        return

    def writeui8(self, *args):
        for x in args:
            self.fp.write(chr(x))
        return

    def writesi8(self, *args):
        for x in args:
            self.fp.write(pack('<b', x))
        return

    def writeui16(self, *args):
        for x in args:
            self.fp.write(pack('<H', x))
        return

    def writeub16(self, *args):
        for x in args:
            self.fp.write(pack('>H', x))
        return

    def writesi16(self, *args):
        for x in args:
            self.fp.write(pack('<h', x))
        return

    def writeub24(self, *args):
        for x in args:
            self.fp.write(pack('>L', x)[1:4])
        return

    def writeui32(self, *args):
        for x in args:
            self.fp.write(pack('<L', x))
        return

    def writeub32(self, *args):
        for x in args:
            self.fp.write(pack('>L', x))
        return

    def writergb(self, xxx_todo_changeme):
        (r, g, b) = xxx_todo_changeme
        self.writeui8(r, g, b)
        return

    def writergba(self, xxx_todo_changeme1):
        (r, g, b, a) = xxx_todo_changeme1
        self.writeui8(r, g, b, a)
        return

    # fixed bits write
    def writebits(self, bits, x, signed=False):
        if signed and x < 0:
            x += (1 << bits)
        assert 0 <= x and x < (1 << bits)
        while True:
            # the number of the remaining bits we can add to the current byte.
            r = 8 - self.bpos
            if bits <= r:
                # |-----8-bits-----|
                # |-bpos-|-bits-|  |
                # |      |----r----|
                self.buff |= x << (r - bits)
                self.bpos += bits               # <= 8
                break
            else:
                # |-----8-bits-----|
                # |-bpos-|---bits----...
                # |      |----r----|
                self.fp.write(chr(self.buff | (x >> (bits - r))))  # r < bits
                self.buff = 0
                self.bpos = 0
                bits -= r                      # cut off the upper r bits
                x &= (1 << bits) - 1
        return

    def finishbits(self):
        if self.bpos:
            self.fp.write(chr(self.buff))
            self.buff = 0
            self.bpos = 0
        return

    # variable length structure

    def writestring(self, s):
        assert '\x00' not in s
        self.write(s)
        self.write('\x00')
        return

    def start_tag(self):
        self.push()
        return


# FLVParser
##
class FLVParser(DataParser):

    TAG_AUDIO = 8
    TAG_VIDEO = 9

    def __init__(self, fp, debug=0):
        DataParser.__init__(self, fp, debug=debug)
        self.tags = []
        self.parse_header()
        self.parse_tags()
        return

    def parse_header(self):
        (F, L, V, ver) = self.read(4)
        assert F + L + V == 'FLV'
        self.flv_version = ord(ver)
        flags = self.readui8()
        self.has_audio = bool(flags & 4)
        self.has_video = bool(flags & 1)
        offset = self.readub32()
        if self.debug:
            print >>sys.stderr, 'Header:', (F, L, V, self.flv_version, flags)
        return

    def parse_metadata(self, data):
        fp = StringIO(data)
        (k, v) = (getvalue(fp), getvalue(fp))
        if self.debug:
            print >>sys.stderr, 'Metadata:', (k, v)
        return (k, v)

    def parse_tags(self):
        try:
            offset = self.readub32()          # always 0
            while True:
                tag = self.readui8()
                length = self.readub24()
                timestamp = self.readub24()     # timestamp in msec.
                reserved = self.readub32()
                offset = self.fp.tell()
                keyframe = False
                if tag == self.TAG_VIDEO and length:
                    keyframe = (self.readui8() & 0x10)
                self.tags.append((tag, length, timestamp, offset, keyframe))
                self.fp.seek(offset + length + 4)  # skip PreviousTagSize
        except EOFError:
            pass
        if self.debug:
            print >>sys.stderr, 'Tags:', len(self.tags)
        return

    def dump(self):
        for (tag, length, timestamp, offset, keyframe) in self.tags:
            print 'tag=%d, length=%d, timestamp=%.03f, keyframe=%r' % (tag, length, timestamp * .001, keyframe)
        return

    def __len__(self):
        return len(self.tags)

    def __iter__(self):
        return iter(self.tags)

    def __getitem__(self, i):
        return self.tags[i]

    def get_duration(self):
        (_, _, duration, _, _) = self.tags[-1]
        return duration

    def get_data(self, i):
        (_, length, _, offset, _) = self.tags[i]
        self.fp.seek(offset)
        data = self.read(length)
        return data

    def seek(self, t):
        i0 = 0
        i1 = len(self.tags)
        while i0 < i1:
            i = (i0 + i1) / 2
            (tag, length, timestamp, offset, keyframe) = self.tags[i]
            if timestamp == t:
                i0 = i
                break
            elif timestamp < t:
                i0 = i
            else:
                i1 = i
        return i0


# FLVWriter
##
# Originally contributed by Luis Fernando <lfkpoa-69@yahoo.com.br>
##
class FLVWriter(DataWriter):

    TAG_AUDIO = 8
    TAG_VIDEO = 9
    TAG_DATA = 18

    def __init__(self, fp, flv_version=1,
                 has_video=True, has_audio=True, has_other=False,
                 framerate=12, debug=0):
        DataWriter.__init__(self, fp, debug=debug)
        self.flv_version = flv_version
        self.has_video = has_video
        self.has_audio = has_audio
        self.has_other = has_other
        self.frames = {}
        self.basetime = 0
        self.duration = 0
        self.metadata = {
            'width': 0, 'height': 0,
            'framerate': framerate, 'duration': 0,
        }
        if self.has_video:
            self.metadata['videocodecid'] = 3
            self.frames[0] = []
        if self.has_audio:
            self.metadata['audiocodecid'] = 2
            self.frames[1] = []
        self.write_header()
        return

    def write_object(self, obj):
        if isinstance(obj, bool):
            self.write('\x01' + chr(obj))
        elif isinstance(obj, (int, long, float)):
            self.write('\x00' + pack('>d', obj))
        elif isinstance(obj, (str, unicode)):
            if isinstance(obj, unicode):
                obj = obj.encode('utf-8')
            if 65535 < len(obj):
                self.write('\x0c' + pack('>L', len(obj)) + obj)
            else:
                self.write('\x02' + pack('>H', len(obj)) + obj)
        elif isinstance(obj, list):
            self.write('\x0a' + pack('>L', len(obj)))
            for x in obj:
                self.write_object(x)
        elif isinstance(obj, dict):
            self.write('\x08' + pack('>L', len(obj)))
            for (k, v) in obj.iteritems():
                assert isinstance(k, str)
                self.write(pack('>H', len(k)) + k)
                self.write_object(v)
        return

    def write_header(self):
        if self.debug:
            print >>sys.stderr, ('write_header: flv_version=%r, audio=%r, video=%r' %
                                 (self.flv_version, self.has_audio, self.has_video))
        self.write('FLV%c' % self.flv_version)
        self.writebits(5, 0)
        self.writebits(1, int(self.has_audio))
        self.writebits(1, 0)
        self.writebits(1, int(self.has_video))
        self.finishbits()
        self.writeub32(9)  # dataoffset (header size) = 9
        self.writeub32(0)  # previous tag size = 0
        self.metadata_pos = self.fp.tell()
        self.write_metadata()
        return

    def write_metadata(self):
        if self.debug:
            print >>sys.stderr, 'write_metadata:', self.metadata
        self.start_tag()
        self.write_object('onMetaData')
        self.write_object(self.metadata)
        self.write('\x00\x00\x09')
        self.end_tag(self.TAG_DATA)
        return

    def end_tag(self, tag, timestamp=None):
        data = self.pop()
        if timestamp is not None:
            self.duration = self.basetime + int(timestamp)
        self.writeui8(tag)
        self.writeub24(len(data))
        self.writeub24(self.duration)
        self.writeui32(0)   # reserved
        self.write(data)
        self.writeub32(len(data) + 11)  # size of this tag
        return

    def write_video_frame(self, timestamp, data):
        if not self.has_video:
            return
        if self.debug:
            print >>sys.stderr, 'write_video_frame: timestamp=%d, data=%d' % (
                timestamp, len(data))
        self.frames[0].append((timestamp, self.TAG_VIDEO, data))
        self._update()
        return

    def write_audio_frame(self, timestamp, data):
        if not self.has_audio:
            return
        if self.debug:
            print >>sys.stderr, 'write_audio_frame: timestamp=%d, data=%d' % (
                timestamp, len(data))
        self.frames[1].append((timestamp, self.TAG_AUDIO, data))
        self._update()
        return

    def write_other_data(self, tag, data):
        if not self.has_other:
            return
        if self.debug:
            print >>sys.stderr, 'write_other_data: tag=%d, data=%d' % (
                tag, len(data))
        self.start_tag()
        self.write(data)
        self.end_tag(tag)
        return

    def _update(self):
        while True:
            frames = None
            for k in sorted(self.frames.iterkeys()):
                v = self.frames[k]
                if not v:
                    return
                if not frames:
                    frames = v
                else:
                    (t0, _, _) = v[0]
                    (t1, _, _) = frames[0]
                    if t0 < t1:
                        frames = v
            (timestamp, tag, data) = frames.pop(0)
            self.start_tag()
            self.write(data)
            self.end_tag(tag, timestamp)
        return

    def set_screen_size(self, width, height):
        if self.debug:
            print >>sys.stderr, 'set_screen_size: %dx%d' % (width, height)
        self.metadata['width'] = width
        self.metadata['height'] = height
        return

    def add_basetime(self, t):
        if self.debug:
            print >>sys.stderr, 'add_basetime: %d+%d' % (self.basetime, t)
        self.basetime += t
        return

    def flush(self):
        if self.debug:
            print >>sys.stderr, 'flush'
        for frames in self.frames.itervalues():
            while frames:
                (timestamp, tag, data) = frames.pop(0)
                self.start_tag()
                self.write(data)
                self.end_tag(tag, timestamp)
        return

    def close(self):
        self.flush()
        DataWriter.close(self)
        # re-write metadata
        self.metadata['duration'] = self.duration * .001
        self.duration = 0
        self.fp.seek(self.metadata_pos)
        self.write_metadata()
        self.fp.flush()
        return


class FlvSpliter():

    TAG_AUDIO = 8
    TAG_VIDEO = 9

    def __init__(self, path):
        self.filename = path
        self.source_fp = open(path, 'rb')
        self.parser = FLVParser(self.source_fp)
        self.__prepare_all_split_points()

    def smooth_split(self, duration, suffix):
        self.__select_best_split_points(duration)
        if self.split_points:
            return self.__split_flv(suffix)
        return None

    def close(self):
        self.source_fp.close()

    def __prepare_all_split_points(self):
        # (tag, length, timestamp, offset, keyframe)
        self.audio_timestamps = [t[2]
                                 for t in self.parser if t[0] == self.TAG_AUDIO]
        self.video_key_timestamps = [
            t[2] for t in self.parser if t[0] == self.TAG_VIDEO and t[4]]
        video_key_ts_aheads = self.___key_video_ahead_audio()
        self.split_points = [(self.video_key_timestamps[i], t) for i, t in
                             enumerate(video_key_ts_aheads)]

    def __select_best_split_points(self, seg_in_sec):
        video_duration_in_sec = max([
            t[2] / 1000 for t in self.parser if t[0] == self.TAG_VIDEO
        ])
        audio_duration_in_sec = max([
            t[2] / 1000 for t in self.parser if t[0] == self.TAG_AUDIO
        ])
        duration_in_sec = max(
            video_duration_in_sec,
            audio_duration_in_sec)
        if 1.5 * seg_in_sec > duration_in_sec:
            print "Video is too short for segmentation in %s sec" % seg_in_sec
            self.split_points = None
        else:
            seg_in_ms = int(1000 * seg_in_sec)
            dur_in_ms = int(
                round(1000 * duration_in_sec))
            seg_times = range(0, dur_in_ms, seg_in_ms)[1:]
            self.split_points = [self.__find_best_split_point(t, seg_in_ms / 3)
                                 for t in seg_times]

    def __find_best_split_point(self, time_in_ms, range):
        # Get all key video timestamps and their delta score
        # from the time region: (time_in_ms - range, time_in_ms + range)
        split_points_in_range = [p for p in self.split_points
                                 if p[0] > time_in_ms - range
                                 and p[0] < time_in_ms + range]
        # If the video track is far shorter than the audio one,
        # there could be nothing in 'split_points_in_range'.
        # In such cases, split will happen on audio track only.
        # Then simply return the closest audio timestamp.
        if not split_points_in_range:
            print "Split point %d is beyond video track" % time_in_ms
            ts_array = np.array(self.audio_timestamps)
            idx = (np.abs(ts_array - time_in_ms)).argmin()
            return self.audio_timestamps[idx]
        else:
            # From the region, find all the video timestamps with min delta, aka
            # the best split points
            deltas = [a[1] for a in split_points_in_range]
            min_delta = min(deltas)
            best_split_points = [p[0]
                                 for p in split_points_in_range if p[1] == min_delta]
            # From all the best splits, find the one closest to the requested
            # time
            ts_array = np.array(best_split_points)
            idx = (np.abs(ts_array - time_in_ms)).argmin()
            print "Split point %d is within video track => %d" % (time_in_ms, best_split_points[idx])
            return best_split_points[idx]

    def ___key_video_ahead_audio(self):
        i = 0
        j = 0
        deltas = []
        while i < len(self.video_key_timestamps):
            while j < len(self.audio_timestamps):
                # For each key video frame, score it with its timestamp difference
                # against the first audio frame behind it
                if self.video_key_timestamps[i] <= self.audio_timestamps[j]:
                    delta = self.audio_timestamps[j] - \
                        self.video_key_timestamps[i]
                    deltas.append(delta)
                    break
                else:
                    j += 1
            else:
                # if the audio track is far shorter than the video track,
                # audio timestamps depletes faster than video ones.  After that,
                # the audio interference against video is none, thus score it
                # with 0.
                deltas.append(0)
            i += 1
        return deltas

    def __generate_split_range(self, split_points):
        # Suppose split_points does not include the 0 and the EOS
        split_points = [0] + split_points + [sys.maxsize]
        split_range = [(split_points[i], split_points[i + 1])
                       for i in range(len(split_points) - 1)]
        return split_range

    def __split_flv(self, suffix):
        parser = self.parser
        split_points = self.split_points
        filename = self.filename

        tag_idx = 0
        audio_header = None
        video_header = None
        split_ranges = self.__generate_split_range(split_points)
        output_files = []

        # Get all sequence headers
        # TODO: More strict check on tags
        while tag_idx < len(parser) and \
            (not audio_header or
             not video_header):
            (tag, _, _, _, keyframe) = parser[tag_idx]
            if tag == self.TAG_AUDIO and not audio_header:
                print "audio header @" + str(tag_idx)
                audio_header = parser.get_data(tag_idx)
            if tag == self.TAG_VIDEO and keyframe and not video_header:
                print "video header @" + str(tag_idx)
                video_header = parser.get_data(tag_idx)
            tag_idx += 1

        for range_idx, split_range in enumerate(split_ranges):
            (start_ts, end_ts) = split_range

            # Generate the segment filename
            seg_idx_str = "%03.d" % range_idx
            segment_filename = os.path.splitext(filename)[0] + '_' + \
                suffix + seg_idx_str + '.flv'
            output_files.append(segment_filename)
            seg_fp = open(segment_filename, 'wb')
            writer = FLVWriter(seg_fp, True)
            print "Split %s [%d, %d) => %s" % \
                (filename, start_ts, end_ts, segment_filename)

            # Reset the timestamp base
            base_timestamp = -1

            # Write audio & video headers
            writer.write_audio_frame(0, audio_header)
            writer.write_video_frame(0, video_header)

            # Main split loop
            while tag_idx < len(parser):
                (tag, _, timestamp, _, _) = parser[tag_idx]
                if timestamp >= start_ts and timestamp < end_ts:
                    base_timestamp = timestamp if base_timestamp < 0 \
                        else base_timestamp
                    timestamp -= base_timestamp
                    if tag == self.TAG_AUDIO:
                        data = parser.get_data(tag_idx)
                        writer.write_audio_frame(timestamp, data)
                    elif tag == self.TAG_VIDEO:
                        data = parser.get_data(tag_idx)
                        writer.write_video_frame(timestamp, data)
                    else:
                        print "Ignore other tag"
                if timestamp >= end_ts:
                    break
                tag_idx += 1
            writer.close()
            seg_fp.close()
        return output_files
    
def main(args):
    filename = args.filename
    duration = args.duration * 60
    suffix = args.suffix
    
    if os.path.exists(filename):
        try:
            parser = FlvSpliter(filename)
            segs = parser.smooth_split(duration, suffix)
            parser.close()
            print "%s => %r" % (filename, segs)
        except AssertionError:
            print "%s is not a valid flash video!!!" % filename
    else:
        print "%s does not exists!!" % filename

import argparse
if __name__ == "__main__":
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Smoothly split flash video without causing Pop-Up noise during segment switch in Adobe Flash Player.')
    parser.add_argument("filename", help="input flash video path")
    parser.add_argument("-d", "--duration",
                        type=int,
                        default=6, 
                        help="expected duration of each segment in minutes (default = 6 minutes).\nPlease be noted this is just a reference for choosing good split points.  So output segments are very likely of different durations.")
    parser.add_argument("-x", "--suffix",
                        default="seg", 
                        help="suffix in each segment's filename (default \"seg\").")
    args = parser.parse_args()
    
    # main logics
    sys.exit(main(args))
        
