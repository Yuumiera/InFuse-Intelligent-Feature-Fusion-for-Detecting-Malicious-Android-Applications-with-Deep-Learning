import zipfile
import struct
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

# Force HAS_ANDROGUARD to False because the original training
# dataset was likely generated using the fallback method.
HAS_ANDROGUARD = False

# ============================================================
# MANIFEST CONSTANTS & UTILS
# ============================================================
DANGEROUS_PERMS = {
    "android.permission.SEND_SMS", "android.permission.READ_SMS",
    "android.permission.RECEIVE_SMS", "android.permission.READ_CONTACTS",
    "android.permission.READ_CALL_LOG", "android.permission.CAMERA",
    "android.permission.RECORD_AUDIO", "android.permission.ACCESS_FINE_LOCATION",
    "android.permission.READ_PHONE_STATE", "android.permission.PROCESS_OUTGOING_CALLS",
    "android.permission.WRITE_EXTERNAL_STORAGE", "android.permission.READ_EXTERNAL_STORAGE",
    "android.permission.GET_ACCOUNTS", "android.permission.USE_CREDENTIALS",
    "android.permission.INSTALL_PACKAGES", "android.permission.DELETE_PACKAGES",
    "android.permission.RECEIVE_BOOT_COMPLETED", "android.permission.BIND_DEVICE_ADMIN",
}

def decode_length(data, pos):
    b = data[pos]
    if b & 0x80:
        length = ((b & 0x7F) << 8) | data[pos + 1]
        return length, pos + 2
    return b, pos + 1

def read_string_table(data, chunk_start):
    strings = []
    try:
        string_count  = struct.unpack_from('<I', data, chunk_start + 8)[0]
        flags         = struct.unpack_from('<I', data, chunk_start + 16)[0]
        strings_start = struct.unpack_from('<I', data, chunk_start + 20)[0]
        is_utf8       = bool(flags & (1 << 8))
        offsets_base  = chunk_start + 28
        data_base     = chunk_start + strings_start

        for i in range(string_count):
            str_off = struct.unpack_from('<I', data, offsets_base + i * 4)[0]
            pos     = data_base + str_off
            try:
                if is_utf8:
                    _, pos = decode_length(data, pos)
                    byte_len, pos = decode_length(data, pos)
                    s = data[pos:pos + byte_len].decode('utf-8', errors='replace')
                else:
                    char_len = struct.unpack_from('<H', data, pos)[0]
                    pos += 2
                    s = data[pos:pos + char_len * 2].decode('utf-16-le', errors='replace')
                strings.append(s)
            except Exception:
                strings.append("")
    except Exception:
        pass
    return strings

def parse_axml(data):
    tags    = []
    strings = []
    if len(data) < 8:
        return tags, True
    pos = 8
    while pos + 8 <= len(data):
        try:
            chunk_type = struct.unpack_from('<H', data, pos)[0]
            chunk_size = struct.unpack_from('<I', data, pos + 4)[0]
        except struct.error:
            break
        if chunk_size < 8 or pos + chunk_size > len(data):
            break
        if chunk_type == 0x0001:
            strings = read_string_table(data, pos)
        elif chunk_type == 0x0102:
            try:
                name_idx   = struct.unpack_from('<I', data, pos + 20)[0]
                attr_count = struct.unpack_from('<H', data, pos + 28)[0]
                tag        = strings[name_idx] if name_idx < len(strings) else ""
                attrs      = {}
                for i in range(attr_count):
                    ap       = pos + 36 + i * 20
                    key_idx  = struct.unpack_from('<I', data, ap + 4)[0]
                    val_idx  = struct.unpack_from('<I', data, ap + 8)[0]
                    val_type = struct.unpack_from('<B', data, ap + 15)[0]
                    val_data = struct.unpack_from('<I', data, ap + 16)[0]
                    key = strings[key_idx] if key_idx < len(strings) else ""
                    if val_type == 0x03:
                        val = strings[val_idx] if val_idx < len(strings) else ""
                    elif val_type == 0x12:
                        val = "true" if val_data else "false"
                    elif val_type == 0x10:
                        val = str(val_data)
                    else:
                        val = str(val_data)
                    if key:
                        attrs[key] = val
                tags.append((tag, attrs))
            except (struct.error, IndexError):
                pass
        pos += chunk_size
    return tags, False

def extract_manifest_features(apk_path: str):
    feats = defaultdict(list)
    feats["package"]        = ""
    feats["exported_count"] = 0
    feats["debuggable"]     = 0

    try:
        with zipfile.ZipFile(apk_path, 'r') as zf:
            if "AndroidManifest.xml" in zf.namelist():
                raw_xml = zf.read("AndroidManifest.xml")
                tags, err = parse_axml(raw_xml)
                
                for tag_name, attrs in tags:
                    name_val = attrs.get("name", "")
                    if tag_name == "manifest":
                        feats["package"] = attrs.get("package", "")
                    elif tag_name == "uses-permission":
                        if name_val: feats["permissions"].append(name_val)
                    elif tag_name == "activity":
                        if name_val: feats["activities"].append(name_val)
                    elif tag_name == "service":
                        if name_val: feats["services"].append(name_val)
                    elif tag_name == "receiver":
                        if name_val: feats["receivers"].append(name_val)
                    elif tag_name == "provider":
                        if name_val: feats["providers"].append(name_val)
                    elif tag_name == "action":
                        if name_val: feats["intent_actions"].append(name_val)
                    elif tag_name == "application":
                        if attrs.get("debuggable", "").lower() == "true":
                            feats["debuggable"] = 1
                    
                    if attrs.get("exported", "").lower() == "true":
                        feats["exported_count"] += 1

                feats["dangerous_perm_count"] = sum(1 for p in feats["permissions"] if p in DANGEROUS_PERMS)
                
                return {
                    "permissions": feats["permissions"],
                    "activities": feats["activities"],
                    "services": feats["services"],
                    "receivers": feats["receivers"],
                    "intent_actions": feats["intent_actions"],
                    "providers": feats["providers"],
                    "num_permissions": len(feats["permissions"]),
                    "num_activities": len(feats["activities"]),
                    "num_services": len(feats["services"]),
                    "num_receivers": len(feats["receivers"]),
                    "num_intent_actions": len(feats["intent_actions"]),
                    "num_providers": len(feats["providers"]),
                    "exported_count": feats["exported_count"],
                    "dangerous_perm_count": feats["dangerous_perm_count"],
                    "debuggable": feats["debuggable"]
                }
    except Exception as e:
        print(f"Error extracting manifest: {e}")
        pass
        
    return {
        "permissions": [], "activities": [], "services": [], "receivers": [], "intent_actions": [], "providers": [],
        "num_permissions": 0, "num_activities": 0, "num_services": 0,
        "num_receivers": 0, "num_intent_actions": 0, "num_providers": 0,
        "exported_count": 0, "dangerous_perm_count": 0, "debuggable": 0
    }

# ============================================================
# STATIC CONSTANTS & UTILS
# ============================================================
DALVIK_OPCODES = [f"op_{i:02x}" for i in range(256)]
SUSPICIOUS_APIS = [
    "sendTextMessage", "sendMultipartTextMessage", "getDeviceId",
    "getSubscriberId", "getLine1Number", "getSimSerialNumber",
    "getNetworkOperator", "getNetworkOperatorName", "getCellLocation",
    "getNeighboringCellInfo", "listenCall", "SmsManager",
    "Runtime;->exec", "ProcessBuilder", "getRuntime",
    "DexClassLoader", "PathClassLoader", "loadClass",
    "defineClass", "loadDex",
    "forName", "getMethod", "getDeclaredMethod", "invoke",
    "getDeclaredField", "setAccessible",
    "Cipher", "SecretKeySpec", "getInstance", "MessageDigest",
    "Mac;->getInstance", "KeyGenerator",
    "HttpURLConnection", "openConnection", "getInputStream",
    "getOutputStream", "Socket", "ServerSocket", "DatagramSocket",
    "URL;->openStream", "HttpClient", "OkHttpClient",
    "Retrofit", "Volley",
    "FileOutputStream", "FileInputStream", "openFileOutput",
    "getExternalStorageDirectory", "MediaStore",
    "getPackageInfo", "getInstalledPackages", "getPackageManager",
    "Build;->MODEL", "Build;->BRAND", "Build;->DEVICE",
    "Build;->FINGERPRINT", "Build;->SERIAL",
    "Camera;->open", "MediaRecorder", "AudioRecord",
    "getLastKnownLocation", "requestLocationUpdates",
    "LocationManager", "getLatitude", "getLongitude",
    "ContactsContract", "CalendarContract", "ContentResolver",
    "AccessibilityService", "DevicePolicyManager",
    "DeviceAdminReceiver", "isAdminActive",
    "startActivity", "startService", "sendBroadcast",
    "registerReceiver", "bindService",
    "SharedPreferences", "getSharedPreferences", "AlarmManager",
    "JobScheduler", "WorkManager",
    "isDebuggerConnected", "Debug;->isDebuggerConnected",
    "getprop", "ro.debuggable", "ro.secure"
]
URL_PATTERN = re.compile(rb'https?://[^\s\x00"\'<>]{5,}', re.IGNORECASE)
IP_PATTERN = re.compile(rb'\b(?:\d{1,3}\.){3}\d{1,3}\b')
EMAIL_PATTERN = re.compile(rb'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}')
BASE64_PATTERN = re.compile(rb'[A-Za-z0-9+/]{20,}={0,2}')
SHELL_CMD_PATTERN = re.compile(rb'(?:chmod|chown|su |mount |remount|/system/bin|/data/local)', re.IGNORECASE)

SUS_KEYWORDS = [
    b"encrypt", b"decrypt", b"cipher", b"AES", b"DES", b"RSA",
    b"base64", b"encode", b"decode", b"obfuscate",
    b"shell", b"root", b"superuser", b"su ",
    b"backdoor", b"exploit", b"payload", b"inject",
    b"keylog", b"screenshot", b"record", b"stealth",
    b"bitcoin", b"wallet", b"ransom", b"locked",
    b"premium", b"subscribe", b"sms", b"send_sms",
    b"password", b"credential", b"login", b"phish",
    b"proxy", b"vpn", b"tunnel", b"tor",
    b"dropper", b"downloader", b"update_url",
    b"c2", b"command", b"control", b"beacon",
    b"emulator", b"genymotion", b"bluestack",
    b"anti_debug", b"anti_vm", b"detect_root",
]

def compute_entropy(data: bytes) -> float:
    if not data: return 0.0
    counter = Counter(data)
    length = len(data)
    entropy = 0.0
    for count in counter.values():
        p = count / length
        if p > 0: entropy -= p * math.log2(p)
    return entropy

def extract_dex_opcode_histogram(dex_bytes: bytes) -> dict:
    features = {op: 0 for op in DALVIK_OPCODES}
    try:
        if HAS_ANDROGUARD:
            try:
                dex = DEX(dex_bytes)
            except:
                dex = DalvikVMFormat(dex_bytes)
            for method in dex.get_methods():
                try:
                    code = method.get_code()
                    if code:
                        for inst in code.get_bc().get_instructions():
                            op_val = inst.get_op_value()
                            if 0 <= op_val < 256: features[f"op_{op_val:02x}"] += 1
                except: pass
        else:
            if len(dex_bytes) > 112:
                data_off = struct.unpack_from('<I', dex_bytes, 104)[0]
                data_size = struct.unpack_from('<I', dex_bytes, 100)[0]
                if data_off + data_size <= len(dex_bytes):
                    code_data = dex_bytes[data_off:data_off + data_size]
                    for byte in code_data[::2]:
                        features[f"op_{byte:02x}"] += 1
    except: pass
    total = sum(features.values())
    if total > 0: features = {k: v / total for k, v in features.items()}
    return features

def extract_api_features(dex_bytes: bytes) -> dict:
    features = {}
    for api in SUSPICIOUS_APIS:
        col = f"api_{api.replace(';->', '_').replace(';', '_')}"
        features[col] = 0
    try:
        for i, api in enumerate(SUSPICIOUS_APIS):
            api_bytes = api.encode('utf-8')
            col = f"api_{api.replace(';->', '_').replace(';', '_')}"
            count = dex_bytes.count(api_bytes)
            features[col] = count
    except: pass

    tel_apis = [a for a in SUSPICIOUS_APIS if any(k in a for k in ["send", "getDevice", "getSub", "getLine", "getSim", "Sms"])]
    net_apis = [a for a in SUSPICIOUS_APIS if any(k in a for k in ["Http", "Socket", "URL", "okhttp", "Retrofit", "Volley", "openConnection"])]
    reflect_apis = [a for a in SUSPICIOUS_APIS if any(k in a for k in ["forName", "getMethod", "getDeclared", "invoke", "setAccessible"])]
    crypto_apis = [a for a in SUSPICIOUS_APIS if any(k in a for k in ["Cipher", "SecretKey", "MessageDigest", "Mac", "KeyGen"])]
    exec_apis = [a for a in SUSPICIOUS_APIS if any(k in a for k in ["exec", "ProcessBuilder", "getRuntime", "DexClassLoader", "PathClass", "loadDex"])]

    features["api_cat_telephony"] = sum(dex_bytes.count(a.encode()) for a in tel_apis)
    features["api_cat_network"] = sum(dex_bytes.count(a.encode()) for a in net_apis)
    features["api_cat_reflection"] = sum(dex_bytes.count(a.encode()) for a in reflect_apis)
    features["api_cat_crypto"] = sum(dex_bytes.count(a.encode()) for a in crypto_apis)
    features["api_cat_exec_dynload"] = sum(dex_bytes.count(a.encode()) for a in exec_apis)

    return features

def extract_file_structure_features(zf: zipfile.ZipFile) -> dict:
    features = {}
    entries = zf.namelist()
    infos = zf.infolist()

    features["file_total_entries"] = len(entries)
    features["file_apk_size"] = sum(i.file_size for i in infos)
    features["file_compressed_size"] = sum(i.compress_size for i in infos)
    ratio = features["file_compressed_size"] / max(features["file_apk_size"], 1)
    features["file_compression_ratio"] = ratio

    exts = Counter(Path(e).suffix.lower() for e in entries)
    features["file_num_dex"] = sum(1 for e in entries if e.endswith('.dex'))
    features["file_num_xml"] = exts.get('.xml', 0)
    features["file_num_png"] = exts.get('.png', 0)
    features["file_num_jpg"] = exts.get('.jpg', 0) + exts.get('.jpeg', 0)
    features["file_num_so"] = exts.get('.so', 0)
    features["file_num_arsc"] = exts.get('.arsc', 0)
    features["file_num_js"] = exts.get('.js', 0)
    features["file_num_html"] = exts.get('.html', 0) + exts.get('.htm', 0)

    features["file_has_assets"] = int(any(e.startswith('assets/') for e in entries))
    features["file_has_lib"] = int(any(e.startswith('lib/') for e in entries))
    features["file_has_res"] = int(any(e.startswith('res/') for e in entries))
    features["file_num_assets"] = sum(1 for e in entries if e.startswith('assets/'))
    features["file_num_res"] = sum(1 for e in entries if e.startswith('res/'))

    dex_entries = [e for e in entries if e.endswith('.dex')]
    if dex_entries:
        try:
            dex_data = zf.read(dex_entries[0])
            features["file_dex_entropy"] = compute_entropy(dex_data)
            features["file_dex_size"] = len(dex_data)
        except:
            features["file_dex_entropy"] = 0.0
            features["file_dex_size"] = 0
    else:
        features["file_dex_entropy"] = 0.0
        features["file_dex_size"] = 0

    features["file_embedded_apk"] = sum(1 for e in entries if e.startswith('assets/') and e.endswith('.apk'))
    features["file_embedded_dex"] = sum(1 for e in entries if e.startswith('assets/') and e.endswith('.dex'))

    return features

def extract_native_lib_features(zf: zipfile.ZipFile) -> dict:
    features = {}
    entries = zf.namelist()

    so_files = [e for e in entries if e.endswith('.so')]
    features["native_has_libs"] = int(len(so_files) > 0)
    features["native_num_libs"] = len(so_files)

    abis = set()
    for so in so_files:
        parts = so.split('/')
        if len(parts) >= 3 and parts[0] == 'lib':
            abis.add(parts[1])

    known_abis = ["armeabi", "armeabi-v7a", "arm64-v8a", "x86", "x86_64", "mips"]
    for abi in known_abis:
        features[f"native_abi_{abi.replace('-', '_')}"] = int(abi in abis)
    features["native_num_abis"] = len(abis)

    so_total_size = 0
    for info in zf.infolist():
        if info.filename.endswith('.so'):
            so_total_size += info.file_size
    features["native_total_size"] = so_total_size

    exec_exts = ('.sh', '.elf', '.bin')
    features["native_exec_in_assets"] = sum(
        1 for e in entries if e.startswith('assets/') and any(e.endswith(ext) for ext in exec_exts)
    )
    return features

def extract_string_features(all_dex_bytes: bytes, zf: zipfile.ZipFile) -> dict:
    features = {}

    urls = URL_PATTERN.findall(all_dex_bytes)
    ips = IP_PATTERN.findall(all_dex_bytes)
    emails = EMAIL_PATTERN.findall(all_dex_bytes)
    base64s = BASE64_PATTERN.findall(all_dex_bytes)
    shell_cmds = SHELL_CMD_PATTERN.findall(all_dex_bytes)

    features["str_num_urls"] = len(urls)
    features["str_num_unique_urls"] = len(set(urls))
    features["str_num_ips"] = len(ips)
    features["str_num_unique_ips"] = len(set(ips))
    features["str_num_emails"] = len(emails)
    features["str_num_base64"] = len(base64s)
    features["str_num_shell_cmds"] = len(shell_cmds)

    http_count = sum(1 for u in urls if u.startswith(b'http://'))
    https_count = sum(1 for u in urls if u.startswith(b'https://'))
    features["str_http_count"] = http_count
    features["str_https_count"] = https_count
    features["str_http_ratio"] = http_count / max(http_count + https_count, 1)

    if urls:
        domain_entropies = []
        for u in urls[:100]:
            try:
                domain = u.split(b'://')[1].split(b'/')[0]
                domain_entropies.append(compute_entropy(domain))
            except: pass
        if domain_entropies:
            features["str_domain_entropy_mean"] = np.mean(domain_entropies)
            features["str_domain_entropy_max"] = np.max(domain_entropies)
        else:
            features["str_domain_entropy_mean"] = 0.0
            features["str_domain_entropy_max"] = 0.0
    else:
        features["str_domain_entropy_mean"] = 0.0
        features["str_domain_entropy_max"] = 0.0

    for kw in SUS_KEYWORDS:
        col = f"str_kw_{kw.decode('utf-8', errors='replace').strip()}"
        features[col] = all_dex_bytes.lower().count(kw.lower())

    asset_entries = [e for e in zf.namelist() if e.startswith('assets/')]
    asset_entropy_vals = []
    for ae in asset_entries[:20]:
        try:
            data = zf.read(ae)
            if len(data) > 0:
                asset_entropy_vals.append(compute_entropy(data))
        except: pass
    features["str_asset_entropy_mean"] = np.mean(asset_entropy_vals) if asset_entropy_vals else 0.0
    features["str_asset_entropy_max"] = np.max(asset_entropy_vals) if asset_entropy_vals else 0.0

    printable_strings = re.findall(rb'[\x20-\x7e]{4,}', all_dex_bytes)
    if printable_strings:
        str_lengths = [len(s) for s in printable_strings]
        features["str_total_count"] = len(printable_strings)
        features["str_len_mean"] = np.mean(str_lengths)
        features["str_len_max"] = max(str_lengths)
        features["str_len_std"] = np.std(str_lengths)
        short_strings = sum(1 for s in printable_strings if len(s) <= 5)
        features["str_short_ratio"] = short_strings / len(printable_strings)
    else:
        features["str_total_count"] = 0
        features["str_len_mean"] = 0.0
        features["str_len_max"] = 0
        features["str_len_std"] = 0.0
        features["str_short_ratio"] = 0.0

    return features

def extract_static_features(apk_path: str):
    features = {}
    try:
        with zipfile.ZipFile(apk_path, 'r') as zf:
            entries = zf.namelist()
            dex_entries = [e for e in entries if e.endswith('.dex')]
            all_dex_bytes = b""
            for de in dex_entries:
                try: all_dex_bytes += zf.read(de)
                except: pass
                
            features.update(extract_dex_opcode_histogram(all_dex_bytes))
            features.update(extract_api_features(all_dex_bytes))
            features.update(extract_file_structure_features(zf))
            features.update(extract_native_lib_features(zf))
            features.update(extract_string_features(all_dex_bytes, zf))
            
    except Exception as e:
        print(f"Error extracting static features: {e}")
    return features

def extract_all(apk_path: str):
    return {
        "manifest": extract_manifest_features(apk_path),
        "static": extract_static_features(apk_path)
    }
