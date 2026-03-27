import cnem2d
print("file:", cnem2d.__file__)

print("\nTest triangolo 0-based...")
try:
    r = cnem2d.SCNI_CNEM2D((0.0,0.0, 1.0,0.0, 0.5,0.866), (3,), (0,1,2))
    print("OK! result[0]:", r[0])
except Exception as e:
    print("EXCEPTION:", e)

print("\nTest triangolo 1-based...")
try:
    r = cnem2d.SCNI_CNEM2D((0.0,0.0, 1.0,0.0, 0.5,0.866), (3,), (1,2,3))
    print("OK! result[0]:", r[0])
except Exception as e:
    print("EXCEPTION:", e)

print("done")
