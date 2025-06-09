from altastata.altastata_functions import AltaStataFunctions
from altastata.altastata_pytorch_dataset import register_altastata_functions_for_pytorch

# Configuration parameters
user_properties = """#My Properties
#Sun Jun 01 19:41:40 EDT 2025
AWSSecretKey=YKPTKMO3GnSr/aJqJpW9QPDtWVrazsnVMvsUNpyHG9usjuaLf4rVt4fmzDtb/8cEPslDxh2AJGYFQntEkr4mDeFXORUxw1XHxxNa/RctYyzxqLTsh7Gaxamm6Bxmy6zG1p+OM78Ykjr736jdx4F0yJJmy1HK19ZuZSmqnkLgsBgqU5l1nEFYsFm5vemQLjC431TfSCbGcVNU1uW/zkwL9U+9KM2rN6HWlmsxqA7t71jslI8Ahf5JWVp1dqvGqf8xgbr5Gw\=\=
media-player=vlcj
myuser=bob123
accounttype=amazon-s3-secure
AWSAccessKeyId=oZP9Iam4mhj0N9uzwzoNT/7xzaj+cWWgIg1MIn5Sgr9zBBahnI04FfcDq2uiYIpVW28H5GKPUZPhnvxnfcvcYZvwAN3oeUnB96o5bg0ABAfizy5r4FbHwxQpFzX0sJTjub0Jv+tvgFR7H/+fO9F8XTfbn/e7WE3n5EKp5nTkyvzcfxU/pef3GN90ut1fMVRLVE47vVINpipto+b2dD0/DwO/SovRz14rvTHYbuIpUPI\=
region=us-east-1
kms-region=us-east-2
metadata-encryption=RSA
acccontainer-prefix=altastata-myorgrsa444-
logging.level.root=WARN
logging.level.com.altastata=WARN
logging.level.org.apache.http=ERROR
logging.level.software.amazon=ERROR"""

private_key = """-----BEGIN RSA PRIVATE KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: DES-EDE3,F26EBECE6DDAEC52

poe21ejZGZQ0GOe+EJjDdJpNvJcq/Yig9aYXY2rCGyxXLGVFeYJFg7z6gMCjIpSd
aprW/0R8L1a2TKbs7f4K5LkSAZ98cd7N45DtIR6B4JFrDGK3LI48/XH3GT3c4OfS
3LYldvy4XeIOAtOTTCoyhN0145ZLSoeEQ7MO3rGK0va3RGLtPWKgeZXH9j5O1Ch4
BvPGMaKapUcgc1slj1GI4Lr+MDSrJKnUNovnVTIClS2rXTEkTri3cPLwcgWjyQIi
BKVnobUD8Gm9irtUD6GeHrkz6Z7ELF3ctSBRSYCg+1FCvRBuljmS2C2aIiE1cu0/
6KcqBnjEPAs250832uhAkZWj5WedIwJv+sJoGJaAUWyOfgG7DHa2HuKeR9KPD2kS
6EygoQtQlXgSvdgZNALtIEfStmnrblTyP9Bh4JU9UzKnE6Tu5h7CjyuzkE0wgIXB
RxgfbURfdDWs22ujLBbWPGfdY+KdNrnmSqxYahKtq6B+99+xuI0GMzX3/rLpOdF0
AGwfa1xNe8/B/Nt+e2FXIhT2xOuH8K3sDn3/FKwy1qIsK+4g5iL6Q0xj07ujkiSI
wZ0X2gtg3L2DW8Y6B8gBdSmDGH+vNX5/CLNn9Ly1VUoMGgs4fUmd3FFZTxiIbpim
rQgQBHP4l1NsSqDrEyplKG83ejloLaVG+hUY1MGv5tF7B1Ta7j8bwoMTmyVCtCrC
P+a7ShdrBUsD2TDhilZhwZcWl0a+FfzR47+faJs/9pSTkyFFp3D4xgKAdME1lvcI
wV5BUmp5CEmbeB4r/+BlFttRZBLBXT1sq80YyQIVLumq0Livao9mOg==
-----END RSA PRIVATE KEY-----"""


# Create an instance of AltaStataFunctions
altastata_functions = AltaStataFunctions.from_credentials(user_properties, private_key)
altastata_functions.set_password("123")

# register the altastata functions with PyTorch-specific registry
register_altastata_functions_for_pytorch(altastata_functions, "bob123_rsa")
