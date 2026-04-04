const { app, BrowserWindow, ipcMain, dialog, shell, systemPreferences } = require('electron');
const path = require('path');
const { spawn, exec } = require('child_process');
const fs = require('fs');
const https = require('https');
const os = require('os');

let mainWindow;
let settingsWindow = null;
let pythonProcess;

function isMacOS() {
  return process.platform === 'darwin';
}

function isWindows() {
  return process.platform === 'win32';
}

function getAppDataDir() {
  if (isMacOS()) {
    return path.join(os.homedir(), 'Library', 'Application Support', 'stenoai');
  }
  if (isWindows()) {
    const appData = process.env.APPDATA || path.join(os.homedir(), 'AppData', 'Roaming');
    return path.join(appData, 'stenoai');
  }
  return path.join(os.homedir(), '.config', 'stenoai');
}

function getResourcesRoot() {
  return path.join(__dirname, '..');
}

function getVenvRoot() {
  return app.isPackaged ? path.join(getAppDataDir(), 'venv') : path.join(getResourcesRoot(), 'venv');
}

function getVenvPythonPath() {
  return isWindows()
    ? path.join(getVenvRoot(), 'Scripts', 'python.exe')
    : path.join(getVenvRoot(), 'bin', 'python');
}

function getPythonEnv() {
  const env = { ...process.env };
  if (app.isPackaged) {
    env.STENOAI_APP_DATA_DIR = getAppDataDir();
  }
  return env;
}

async function findPythonCommand() {
  const candidates = isWindows()
    ? [
        { command: 'py', args: ['-3.12', '--version'] },
        { command: 'py', args: ['-3.11', '--version'] },
        { command: 'py', args: ['-3.10', '--version'] },
        { command: 'py', args: ['-3.9', '--version'] },
        { command: 'py', args: ['-3', '--version'] },
        { command: 'python', args: ['--version'] },
        { command: 'python3', args: ['--version'] }
      ]
    : [
        { command: 'python3', args: ['--version'] },
        { command: 'python', args: ['--version'] }
      ];

  for (const candidate of candidates) {
    const found = await new Promise((resolve) => {
      const proc = spawn(candidate.command, candidate.args, { stdio: 'ignore' });
      proc.on('error', () => resolve(false));
      proc.on('close', (code) => resolve(code === 0));
    });

    if (found) {
      return candidate;
    }
  }

  return null;
}

function getFfmpegCandidates() {
  const candidates = ['ffmpeg'];

  if (isMacOS()) {
    candidates.push('/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg', '/usr/bin/ffmpeg');
  } else if (isWindows()) {
    candidates.push(
      'C:\\ffmpeg\\bin\\ffmpeg.exe',
      'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe',
      'C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe'
    );
  } else {
    candidates.push('/usr/local/bin/ffmpeg', '/usr/bin/ffmpeg');
  }

  return candidates;
}

/**
 * Validate that a file path is within allowed directories (security)
 * Prevents path traversal attacks by ensuring files are only accessed
 * within the app's designated data directories
 */
function validateSafeFilePath(filepath, allowedBaseDirs) {
  if (!filepath) return false;

  try {
    // Resolve to absolute path and normalize
    const resolvedPath = path.resolve(filepath);

    // Ensure it's within one of the allowed base directories
    for (const baseDir of allowedBaseDirs) {
      const resolvedBase = path.resolve(baseDir);
      if (resolvedPath.startsWith(resolvedBase + path.sep) || resolvedPath === resolvedBase) {
        return true;
      }
    }

    return false;
  } catch (error) {
    console.error('Error validating file path:', error);
    return false;
  }
}

function createWindow() {
  const windowOptions = {
    width: 1200,
    height: 800,
    minWidth: 1000,
    minHeight: 600,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    },
    show: false
  };

  if (isMacOS()) {
    windowOptions.titleBarStyle = 'hiddenInset';
  }

  mainWindow = new BrowserWindow(windowOptions);

  mainWindow.loadFile('index.html');
  
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
    if (pythonProcess) {
      pythonProcess.kill();
    }
  });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

function createSettingsWindow() {
  if (settingsWindow) {
    settingsWindow.focus();
    return;
  }

  const settingsOptions = {
    width: 900,
    height: 700,
    minWidth: 800,
    minHeight: 600,
    parent: mainWindow,
    modal: false,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    },
    show: false,
    backgroundColor: '#1a1a1a'
  };

  if (isMacOS()) {
    settingsOptions.titleBarStyle = 'hiddenInset';
  }

  settingsWindow = new BrowserWindow(settingsOptions);

  settingsWindow.loadFile(path.join(__dirname, 'settings.html'));

  settingsWindow.once('ready-to-show', () => {
    settingsWindow.show();
  });

  settingsWindow.on('closed', () => {
    settingsWindow = null;
  });
}


// Microphone permission handlers
ipcMain.handle('check-microphone-permission', async () => {
  try {
    if (!isMacOS()) {
      return { success: true, status: 'granted' };
    }
    const status = systemPreferences.getMediaAccessStatus('microphone');
    console.log('Microphone permission status:', status);
    return { success: true, status };
  } catch (error) {
    console.error('Error checking microphone permission:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('request-microphone-permission', async () => {
  try {
    if (!isMacOS()) {
      return { success: true, granted: true };
    }
    console.log('Requesting microphone permission...');
    const granted = await systemPreferences.askForMediaAccess('microphone');
    console.log('Microphone permission granted:', granted);
    return { success: true, granted };
  } catch (error) {
    console.error('Error requesting microphone permission:', error);
    return { success: false, error: error.message };
  }
});

// IPC handler for opening settings
ipcMain.handle('open-settings', () => {
  createSettingsWindow();
});

// Debug functionality handled by side panel now

// Python backend communication
function runPythonScript(script, args = [], silent = false) {
  return new Promise((resolve, reject) => {
    const pythonPath = getVenvPythonPath();
    const scriptPath = path.join(getResourcesRoot(), script);

    // Log the command being executed (unless silent)
    const command = `${pythonPath} ${scriptPath} ${args.join(' ')}`;
    console.log('Running:', command);
    if (!silent) {
      sendDebugLog(`$ ${script} ${args.join(' ')}`);
    }

    const process = spawn(pythonPath, ['-u', scriptPath, ...args], {
      cwd: getResourcesRoot(),
      env: getPythonEnv()
    });

    let stdout = '';
    let stderr = '';

    process.stdout.on('data', (data) => {
      const output = data.toString();
      stdout += output;
      console.log('Python stdout:', output);
      // Stream stdout to debug panel in real-time (unless silent)
      if (!silent) {
        output.split('\n').forEach(line => {
          if (line.trim()) sendDebugLog(line.trim());
        });
      }
    });

    process.stderr.on('data', (data) => {
      const output = data.toString();
      stderr += output;
      console.log('Python stderr:', output);
      // Stream stderr to debug panel in real-time (unless silent)
      if (!silent) {
        output.split('\n').forEach(line => {
          if (line.trim()) sendDebugLog('STDERR: ' + line.trim());
        });
      }
    });

    process.on('close', (code) => {
      if (!silent) {
        sendDebugLog(`Command completed with exit code: ${code}`);
      }
      if (code === 0) {
        resolve(stdout);
      } else {
        reject(new Error(`Python script failed with code ${code}: ${stderr}`));
      }
    });
    
    process.on('error', (error) => {
      sendDebugLog(`Command error: ${error.message}`);
      reject(error);
    });
  });
}

// IPC Handlers - Separate start/stop with better error handling
ipcMain.handle('start-recording', async (event, sessionName) => {
  try {
    sendDebugLog(`Starting recording session: ${sessionName || 'Meeting'}`);
    sendDebugLog('$ python simple_recorder.py start');
    
    // Start recording (removed clear-state to prevent race conditions)
    const result = await runPythonScript('simple_recorder.py', ['start', sessionName || 'Meeting']);
    
    if (result.includes('SUCCESS')) {
      sendDebugLog('Recording started successfully');
      return { success: true, message: result };
    } else {
      sendDebugLog(`Recording failed: ${result}`);
      return { success: false, error: result };
    }
  } catch (error) {
    console.error('Start recording error:', error.message);
    sendDebugLog(`Recording error: ${error.message}`);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('stop-recording', async () => {
  try {
    const result = await runPythonScript('simple_recorder.py', ['stop']);
    
    if (result.includes('SUCCESS') || result.includes('Recording saved')) {
      return { success: true, message: result };
    } else {
      return { success: false, error: result };
    }
  } catch (error) {
    console.error('Stop recording error:', error.message);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('get-status', async () => {
  try {
    const result = await runPythonScript('simple_recorder.py', ['status'], true); // Silent mode
    return { success: true, status: result };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('process-recording', async (event, audioFile, sessionName) => {
  try {
    const result = await runPythonScript('simple_recorder.py', ['process', audioFile, '--name', sessionName]);
    return { success: true, result: result };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('test-system', async () => {
  try {
    const result = await runPythonScript('simple_recorder.py', ['test']);
    return { success: true, result: result };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('select-audio-file', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Audio Files', extensions: ['wav', 'mp3', 'm4a', 'aac'] }
    ]
  });
  
  if (!result.canceled && result.filePaths.length > 0) {
    return { success: true, filePath: result.filePaths[0] };
  }
  
  return { success: false, error: 'No file selected' };
});

ipcMain.handle('list-meetings', async () => {
  try {
    const result = await runPythonScript('simple_recorder.py', ['list-meetings']);
    return { success: true, meetings: JSON.parse(result) };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('clear-state', async () => {
  try {
    const result = await runPythonScript('simple_recorder.py', ['clear-state']);
    return { success: true, message: result };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('update-meeting', async (event, meetingFilePath, updates) => {
  try {
    const projectRoot = getResourcesRoot();

    // Define allowed base directories for file operations
    const allowedBaseDirs = [
      projectRoot,
      getAppDataDir()
    ];

    // Convert to absolute path if needed
    const absolutePath = path.isAbsolute(meetingFilePath)
      ? meetingFilePath
      : path.join(projectRoot, meetingFilePath);

    // Security: Validate file path is within allowed directories
    if (!validateSafeFilePath(absolutePath, allowedBaseDirs)) {
      console.error(`Security: Blocked attempt to update file outside allowed directories: ${absolutePath}`);
      return {
        success: false,
        error: 'Invalid file path'
      };
    }

    // Read existing data
    if (!fs.existsSync(absolutePath)) {
      return {
        success: false,
        error: 'Meeting file not found'
      };
    }

    const data = JSON.parse(fs.readFileSync(absolutePath, 'utf8'));

    // Update fields - only update fields that are provided
    if (updates.name !== undefined) {
      data.session_info.name = updates.name;
    }
    if (updates.summary !== undefined) {
      data.summary = updates.summary;
    }
    if (updates.participants !== undefined) {
      data.participants = updates.participants;
    }
    if (updates.key_points !== undefined) {
      data.key_points = updates.key_points;
    }
    if (updates.action_items !== undefined) {
      data.action_items = updates.action_items;
    }

    // Add updated timestamp
    data.session_info.updated_at = new Date().toISOString();

    // Write back to file
    fs.writeFileSync(absolutePath, JSON.stringify(data, null, 2), 'utf8');

    console.log(`Updated meeting: ${absolutePath}`);

    return {
      success: true,
      message: 'Meeting updated successfully',
      updatedData: data
    };
  } catch (error) {
    console.error('Update meeting error:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('delete-meeting', async (event, meetingData) => {
  try {
    const fs = require('fs');
    const path = require('path');

    // meetingData is the actual meeting object, not a file path
    const meeting = meetingData;

    // Build correct file paths from the meeting data - convert to absolute paths
    const projectRoot = getResourcesRoot();

    // Define allowed base directories for file operations
    const allowedBaseDirs = [
      projectRoot,
      getAppDataDir()
    ];

    const meetingFile = meeting.session_info?.meeting_file || meeting.session_info?.summary_file;
    const transcriptFile = meeting.session_info?.transcript_file;

    // Convert relative paths to absolute paths
    const absolutePaths = [];
    if (meetingFile) {
      absolutePaths.push(path.isAbsolute(meetingFile) ? meetingFile : path.join(projectRoot, meetingFile));
    }
    if (transcriptFile) {
      absolutePaths.push(path.isAbsolute(transcriptFile) ? transcriptFile : path.join(projectRoot, transcriptFile));
    }

    console.log('Attempting to delete files:', absolutePaths);

    let deletedCount = 0;
    let validationErrors = 0;

    // Delete all related files with path validation
    for (const file of absolutePaths) {
      try {
        // Security: Validate file path is within allowed directories
        if (!validateSafeFilePath(file, allowedBaseDirs)) {
          console.error(`Security: Blocked attempt to delete file outside allowed directories: ${file}`);
          validationErrors++;
          continue;
        }

        if (fs.existsSync(file)) {
          fs.unlinkSync(file);
          deletedCount++;
          console.log(`Deleted: ${file}`);
        } else {
          console.log(`File not found (already deleted?): ${file}`);
        }
      } catch (err) {
        console.warn(`Could not delete ${file}:`, err.message);
      }
    }

    if (validationErrors > 0) {
      return {
        success: false,
        error: `Blocked ${validationErrors} file deletion(s) due to security validation`
      };
    }
    
    return { 
      success: true, 
      message: `Deleted meeting and ${deletedCount} associated files` 
    };
  } catch (error) {
    console.error('Delete meeting error:', error);
    return { success: false, error: error.message };
  }
});

// Queue status handler
ipcMain.handle('get-queue-status', async () => {
  return {
    success: true,
    isProcessing,
    queueSize: processingQueue.length,
    currentJob: currentProcessingJob?.sessionName || null,
    hasRecording: currentRecordingProcess !== null
  };
});

// Global recording state management
let currentRecordingProcess = null;
let currentCaptureMode = null;
let processingQueue = [];
let isProcessing = false;
let currentProcessingJob = null;

function broadcastCaptureMode(mode) {
  currentCaptureMode = mode;

  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('capture-mode-changed', { mode });
  }
}

// Processing queue management
async function processNextInQueue() {
  if (isProcessing || processingQueue.length === 0) {
    return;
  }
  
  isProcessing = true;
  currentProcessingJob = processingQueue.shift();
  
  console.log(`🔄 Processing queued job: ${currentProcessingJob.sessionName}`);
  
  try {
    const result = await runPythonScript('simple_recorder.py', ['process', currentProcessingJob.audioFile, '--name', currentProcessingJob.sessionName]);
    console.log(`✅ Completed processing: ${currentProcessingJob.sessionName}`);
    
    // Notify frontend about completion with processed meeting data
    if (mainWindow) {
      try {
        // Get the specific processed meeting data
        const meetingsResult = await runPythonScript('simple_recorder.py', ['list-meetings']);
        const allMeetings = JSON.parse(meetingsResult);
        const processedMeeting = allMeetings.find(m => m.session_info?.name === currentProcessingJob.sessionName);
        
        mainWindow.webContents.send('processing-complete', { 
          success: true, 
          sessionName: currentProcessingJob.sessionName,
          message: 'Processing completed successfully',
          meetingData: processedMeeting
        });
      } catch (error) {
        console.error('Error getting processed meeting data:', error);
        mainWindow.webContents.send('processing-complete', { 
          success: true, 
          sessionName: currentProcessingJob.sessionName,
          message: 'Processing completed successfully'
        });
      }
    }
    
  } catch (error) {
    console.error(`❌ Processing failed for ${currentProcessingJob.sessionName}:`, error);
    
    // Notify frontend about failure
    if (mainWindow) {
      mainWindow.webContents.send('processing-complete', { 
        success: false, 
        sessionName: currentProcessingJob.sessionName,
        error: error.message
      });
    }
  } finally {
    isProcessing = false;
    currentProcessingJob = null;
    // Process next job in queue
    setTimeout(processNextInQueue, 1000);
  }
}

function addToProcessingQueue(audioFile, sessionName) {
  processingQueue.push({ audioFile, sessionName });
  console.log(`📋 Added to processing queue: ${sessionName} (Queue size: ${processingQueue.length})`);
  processNextInQueue();
}

ipcMain.handle('start-recording-ui', async (_, sessionName) => {
  try {
    if (currentRecordingProcess) {
      return { success: false, error: 'Recording already in progress' };
    }

    // Start recording (removed clear-state to prevent race conditions)
    
    console.log('Starting long recording process...');
    sendDebugLog(`Starting recording process: ${sessionName || 'Meeting'}`);
    sendDebugLog('$ python simple_recorder.py record 3600');
    
    const pythonPath = getVenvPythonPath();
    const scriptPath = path.join(getResourcesRoot(), 'simple_recorder.py');
    
    const actualSessionName = sessionName || 'Meeting';
    let stdoutBuffer = '';
    
    // Start background recording with 60-minute limit
    currentRecordingProcess = spawn(pythonPath, ['-u', scriptPath, 'record', '3600', actualSessionName], {
      cwd: getResourcesRoot(),
      env: getPythonEnv()
    });
    broadcastCaptureMode('starting');

    let hasStarted = false;
    
    currentRecordingProcess.stdout.on('data', (data) => {
      const output = data.toString();
      console.log('Recording stdout:', output);

      stdoutBuffer += output;
      const lines = stdoutBuffer.split(/\r?\n/);
      stdoutBuffer = lines.pop();

      lines.forEach((rawLine) => {
        const line = rawLine.trim();
        if (!line) {
          return;
        }

        sendDebugLog(line);

        if (line.startsWith('CAPTURE_MODE:')) {
          const mode = line.split(':').slice(1).join(':').trim() || null;
          broadcastCaptureMode(mode);
        }

        if (line.includes('✅ Complete processing finished!')) {
          console.log(`🎉 Recording and processing completed for: ${actualSessionName}`);
          if (mainWindow) {
            runPythonScript('simple_recorder.py', ['list-meetings'])
              .then(meetingsResult => {
                const allMeetings = JSON.parse(meetingsResult);
                const processedMeeting = allMeetings.find(m => m.session_info?.name === actualSessionName);
                
                mainWindow.webContents.send('processing-complete', { 
                  success: true, 
                  sessionName: actualSessionName,
                  message: 'Recording and processing completed successfully',
                  meetingData: processedMeeting
                });
              })
              .catch(error => {
                console.error('Error getting processed meeting data:', error);
                mainWindow.webContents.send('processing-complete', { 
                  success: true, 
                  sessionName: actualSessionName,
                  message: 'Recording and processing completed successfully'
                });
              });
          }
        }

        if (line.includes('Recording to:') && !hasStarted) {
          hasStarted = true;
        }
      });
    });

    currentRecordingProcess.stderr.on('data', (data) => {
      const output = data.toString();
      console.log('Recording stderr:', output);

      // Send real-time stderr to debug panel (same as runPythonScript)
      output.split('\n').forEach(line => {
        if (line.trim()) {
          sendDebugLog('STDERR: ' + line.trim());

          // Parse real-time transcript segments from log output
          // Format: "2026-01-10 00:08:36,042 - INFO - [system] Other: text here"
          // or: "2026-01-10 00:09:01,121 - INFO - [microphone] You: text here"
          const transcriptMatch = line.match(/\[(?:system|microphone)\]\s*(You|Other):\s*(.+)/);
          if (transcriptMatch && mainWindow && !mainWindow.isDestroyed()) {
            const speaker = transcriptMatch[1];
            const text = transcriptMatch[2];
            const timestamp = new Date().toLocaleTimeString();
            mainWindow.webContents.send('realtime-transcript', {
              speaker: speaker,
              text: text,
              timestamp: timestamp
            });
          }
        }
      });
    });

    currentRecordingProcess.on('close', (code) => {
      console.log(`Recording process closed with code ${code}`);
      sendDebugLog(`Recording process completed with exit code: ${code}`);
      currentRecordingProcess = null;
      broadcastCaptureMode(null);
    });

    // Give it time to start
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    if (currentRecordingProcess) {
      return { success: true, message: 'Recording started successfully' };
    } else {
      return { success: false, error: 'Failed to start recording process' };
    }
  } catch (error) {
    console.error('Start recording UI error:', error.message);
    currentRecordingProcess = null;
    return { success: false, error: error.message };
  }
});

ipcMain.handle('stop-recording-ui', async () => {
  try {
    if (!currentRecordingProcess) {
      return { success: false, error: 'No recording in progress' };
    }

    console.log('Stopping recording process...');
    broadcastCaptureMode(null);

    // Request stop through the backend so both macOS and Windows can finish gracefully.
    await runPythonScript('simple_recorder.py', ['stop'], true);
    
    return { 
      success: true, 
      message: 'Recording stopped - processing will complete in background'
    };
  } catch (error) {
    console.error('Stop recording UI error:', error.message);
    currentRecordingProcess = null;
    return { success: false, error: error.message };
  }
});

// Setup IPC handlers

ipcMain.handle('startup-setup-check', async () => {
  try {
    console.log('Running startup setup check...');
    
    // Use Python backend to check setup
    const result = await runPythonScript('simple_recorder.py', ['setup-check']);
    console.log('Setup check result:', result);
    
    // Parse the output to determine if setup is complete
    const allGood = result.includes('🎉 System check passed!');
    
    // Extract check results for UI display
    const lines = result.split('\n');
    const checks = [];
    
    lines.forEach(line => {
      if (line.includes('✅') || line.includes('❌') || line.includes('⚠️')) {
        const parts = line.split(/\s{2,}/); // Split on multiple spaces
        if (parts.length >= 2) {
          checks.push([parts[0].trim(), parts[1].trim()]);
        }
      }
    });
    
    console.log('Parsed checks:', checks);
    console.log('All good:', allGood);
    
    return { 
      success: true, 
      allGood,
      checks
    };
  } catch (error) {
    console.error('Setup check error:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('setup-system-check', async () => {
  try {
    const pythonCommand = await findPythonCommand();
    if (!pythonCommand) {
      return { success: false, error: 'Python 3 not found. Please install Python 3.8+' };
    }
    
    // Create required directories - match Python logic for DMG vs development
    const baseDir = app.isPackaged ? getAppDataDir() : path.join(__dirname, '..');
    
    const dirs = ['recordings', 'transcripts', 'output'];
    
    for (const dir of dirs) {
      const dirPath = path.join(baseDir, dir);
      if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
      }
    }
    
    // Create venv directory if it doesn't exist  
    const venvPath = getVenvRoot();
    if (!fs.existsSync(venvPath)) {
      await new Promise((resolve, reject) => {
        const process = spawn(
          pythonCommand.command,
          [...pythonCommand.args.filter(arg => arg !== '--version'), '-m', 'venv', 'venv'],
          {
            cwd: baseDir
          }
        );
        
        process.on('close', (code) => {
          if (code === 0) {
            resolve();
          } else {
            reject(new Error('Failed to create virtual environment'));
          }
        });
        
        process.on('error', reject);
      });
    }
    
    return { success: true, message: 'System setup complete - Python and directories ready' };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('setup-ffmpeg', async () => {
  try {
    sendDebugLog('$ Checking for existing ffmpeg installation...');
    sendDebugLog(`$ Checking candidates: ${getFfmpegCandidates().join(', ')}`);

    // Check if ffmpeg is already installed - try multiple common paths
    const ffmpegPaths = getFfmpegCandidates();
    let ffmpegPath = null;

    for (const testPath of ffmpegPaths) {
      try {
        const found = await new Promise((resolve) => {
          const proc = spawn(testPath, ['-version'], { timeout: 5000 });
          proc.on('error', () => resolve(false));
          proc.on('close', (code) => resolve(code === 0));
        });

        if (found) {
          ffmpegPath = testPath;
          sendDebugLog(`Found ffmpeg at: ${testPath}`);
          break;
        }
      } catch (error) {
        // Try next path
        continue;
      }
    }

    if (!ffmpegPath) {
      sendDebugLog('ffmpeg not found in any common locations');
    }
    
    // Install ffmpeg if not present
    if (!ffmpegPath) {
      if (isWindows()) {
        const installers = [
          { cmd: 'winget', args: ['install', '--id', 'Gyan.FFmpeg', '--accept-package-agreements', '--accept-source-agreements'] },
          { cmd: 'choco', args: ['install', 'ffmpeg', '-y'] },
          { cmd: 'scoop', args: ['install', 'ffmpeg'] }
        ];

        let installed = false;
        for (const installer of installers) {
          const available = await new Promise((resolve) => {
            const proc = spawn(installer.cmd, ['--version'], { stdio: 'ignore' });
            proc.on('error', () => resolve(false));
            proc.on('close', (code) => resolve(code === 0));
          });

          if (!available) {
            continue;
          }

          sendDebugLog(`$ ${installer.cmd} ${installer.args.join(' ')}`);
          installed = await new Promise((resolve) => {
            const proc = spawn(installer.cmd, installer.args, { stdio: 'pipe' });
            proc.stdout.on('data', (data) => sendDebugLog(data.toString().trim()));
            proc.stderr.on('data', (data) => sendDebugLog('STDERR: ' + data.toString().trim()));
            proc.on('error', () => resolve(false));
            proc.on('close', (code) => resolve(code === 0));
          });

          if (installed) {
            sendDebugLog(`ffmpeg installation completed successfully via ${installer.cmd}`);
            break;
          }
        }

        if (!installed) {
          return { success: false, error: 'ffmpeg not found. Install it manually or use winget/choco/scoop.' };
        }
      } else {
        sendDebugLog('ffmpeg not found, checking for Homebrew...');
        sendDebugLog('$ Checking: brew, /opt/homebrew/bin/brew, /usr/local/bin/brew');

        const brewPaths = ['brew', '/opt/homebrew/bin/brew', '/usr/local/bin/brew'];
        let brewPath = null;

        for (const testPath of brewPaths) {
          try {
            const found = await new Promise((resolve) => {
              const proc = spawn(testPath, ['--version'], { timeout: 5000 });
              proc.on('error', () => resolve(false));
              proc.on('close', (code) => resolve(code === 0));
            });

            if (found) {
              brewPath = testPath;
              sendDebugLog(`Found Homebrew at: ${testPath}`);
              break;
            }
          } catch (error) {
            continue;
          }
        }

        if (!brewPath) {
          return { success: false, error: 'ffmpeg not found and Homebrew is unavailable.' };
        }

        sendDebugLog(`$ ${brewPath} install ffmpeg`);
        await new Promise((resolve, reject) => {
          const process = spawn(brewPath, ['install', 'ffmpeg'], { timeout: 300000 });

          process.stdout.on('data', (data) => {
            sendDebugLog(data.toString().trim());
          });

          process.stderr.on('data', (data) => {
            sendDebugLog('STDERR: ' + data.toString().trim());
          });

          process.on('close', (code) => {
            if (code === 0) {
              sendDebugLog('ffmpeg installation completed successfully');
              resolve();
            } else {
              sendDebugLog(`ffmpeg installation failed with exit code: ${code}`);
              reject(new Error('Failed to install ffmpeg via Homebrew'));
            }
          });

          process.on('error', (error) => {
            sendDebugLog(`ffmpeg installation error: ${error.message}`);
            reject(error);
          });
        });
      }
    } else {
      sendDebugLog('ffmpeg already installed, skipping installation');
    }
    
    return { success: true, message: 'ffmpeg ready' };
  } catch (error) {
    sendDebugLog(`ffmpeg setup failed: ${error.message}`);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('setup-python', async () => {
  try {
    const projectRoot = getResourcesRoot();
    const venvPath = getVenvRoot();
    const pythonCommand = await findPythonCommand();

    if (!pythonCommand) {
      return { success: false, error: 'Python 3 not found. Please install Python 3.8+' };
    }
    
    sendDebugLog(`Resources directory: ${projectRoot}`);
    sendDebugLog(`Virtualenv directory: ${venvPath}`);
    
    // Create virtual environment if it doesn't exist
    if (!fs.existsSync(venvPath)) {
      sendDebugLog('Python virtual environment not found, creating...');
      sendDebugLog(`$ ${pythonCommand.command} ${[...pythonCommand.args.filter(arg => arg !== '--version'), '-m', 'venv', 'venv'].join(' ')}`);
      
      await new Promise((resolve, reject) => {
        const process = spawn(
          pythonCommand.command,
          [...pythonCommand.args.filter(arg => arg !== '--version'), '-m', 'venv', 'venv'],
          {
            cwd: path.dirname(venvPath),
            stdio: 'pipe'
          }
        );
        
        process.stdout.on('data', (data) => {
          sendDebugLog(data.toString().trim());
        });
        
        process.stderr.on('data', (data) => {
          sendDebugLog('STDERR: ' + data.toString().trim());
        });
        
        process.on('close', (code) => {
          if (code === 0) {
            sendDebugLog('Virtual environment created successfully');
            resolve();
          } else {
            sendDebugLog(`Virtual environment creation failed with exit code: ${code}`);
            reject(new Error('Failed to create virtual environment'));
          }
        });
        
        process.on('error', (error) => {
          sendDebugLog(`Process error: ${error.message}`);
          reject(error);
        });
      });
    } else {
      sendDebugLog('Python virtual environment already exists');
    }
    
    // Install the app's Python requirements.
    sendDebugLog('Installing Python dependencies...');
    sendDebugLog('$ pip install -r requirements.txt');
    
    return new Promise((resolve) => {
      const pythonPath = getVenvPythonPath();
      const requirementsPath = path.join(projectRoot, 'requirements.txt');
      const process = spawn(pythonPath, ['-m', 'pip', 'install', '-r', requirementsPath], {
        cwd: projectRoot,
        stdio: 'pipe'
      });
      
      let output = '';
      
      process.stdout.on('data', (data) => {
        const text = data.toString().trim();
        if (text) {
          sendDebugLog(text);
          output += text;
        }
      });
      
      process.stderr.on('data', (data) => {
        const text = data.toString().trim();
        if (text) {
          sendDebugLog('STDERR: ' + text);
          output += text;
        }
      });
      
      process.on('close', (code) => {
        if (code === 0) {
          sendDebugLog('Python dependencies installation completed successfully');
          resolve({ success: true, message: 'Python dependencies installed' });
        } else {
          sendDebugLog(`Python dependencies installation failed with exit code: ${code}`);
          resolve({ success: false, error: `Installation failed: ${output}` });
        }
      });
      
      process.on('error', (error) => {
        resolve({ success: false, error: `Process error: ${error.message}` });
      });
    });
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// Add IPC handler for sending debug logs to frontend
function sendDebugLog(message) {
  // Send to main window (both setup console and debug panel)
  if (mainWindow) {
    mainWindow.webContents.send('debug-log', message);
  }
}

ipcMain.handle('setup-whisper', async () => {
  try {
    const projectRoot = getResourcesRoot();
    const pythonPath = getVenvPythonPath();
    
    sendDebugLog('Installing Whisper speech recognition...');
    sendDebugLog(`$ ${pythonPath} -m pip install openai-whisper`);
    
    return new Promise((resolve) => {
      const process = spawn(pythonPath, ['-m', 'pip', 'install', 'openai-whisper'], {
        cwd: projectRoot,
        stdio: 'pipe'
      });
      
      let output = '';
      
      process.stdout.on('data', (data) => {
        const text = data.toString().trim();
        if (text) {
          sendDebugLog(text);
          output += text;
        }
      });
      
      process.stderr.on('data', (data) => {
        const text = data.toString().trim();
        if (text) {
          sendDebugLog('STDERR: ' + text);
          output += text;
        }
      });
      
      process.on('close', (code) => {
        if (code === 0) {
          sendDebugLog('Whisper installation completed successfully');
          resolve({ success: true, message: 'Whisper installed successfully' });
        } else {
          sendDebugLog(`Whisper installation failed with exit code: ${code}`);
          resolve({ success: false, error: `Whisper installation failed: ${output}` });
        }
      });
      
      process.on('error', (error) => {
        resolve({ success: false, error: `Process error: ${error.message}` });
      });
    });
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('setup-test', async () => {
  try {
    sendDebugLog('Running system test...');
    sendDebugLog('$ python simple_recorder.py test');
    
    // Test the complete system
    const result = await runPythonScript('simple_recorder.py', ['test']);
    
    // Log the full result to debug console
    result.split('\n').forEach(line => {
      if (line.trim()) sendDebugLog(line.trim());
    });
    
    if (result.includes('System check passed') || result.includes('SUCCESS')) {
      sendDebugLog('System test completed successfully');
      return { success: true, message: 'System test passed' };
    } else {
      // Extract specific error details from the output
      const errorLines = result.split('\n').filter(line => line.includes('ERROR:'));
      const specificError = errorLines.length > 0 ? errorLines[errorLines.length - 1].replace('ERROR: ', '') : 'Unknown error';
      sendDebugLog(`System test failed: ${specificError}`);
      return { success: false, error: `System test failed: ${specificError}`, details: result };
    }
  } catch (error) {
    sendDebugLog(`System test error: ${error.message}`);
    return { success: false, error: error.message };
  }
});

// Settings window IPC handlers  
ipcMain.handle('trigger-setup-wizard', async () => {
  try {
    console.log('🔧 Starting setup wizard from settings...');
    
    // Trigger the main window's setup flow
    if (mainWindow) {
      mainWindow.webContents.send('trigger-setup-flow');
    }
    
    return { success: true, message: 'Setup wizard triggered in main window' };
  } catch (error) {
    console.error('Setup wizard failed:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('get-app-version', async () => {
  try {
    const packagePath = path.join(__dirname, 'package.json');
    const packageContent = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
    return {
      success: true,
      version: packageContent.version,
      name: packageContent.productName || packageContent.name
    };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('get-notifications', async () => {
  try {
    const result = await runPythonScript('simple_recorder.py', ['get-notifications']);
    const jsonData = JSON.parse(result);

    return {
      success: true,
      ...jsonData
    };
  } catch (error) {
    sendDebugLog(`Error getting notification settings: ${error.message}`);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('set-notifications', async (event, enabled) => {
  try {
    sendDebugLog(`Setting notifications to: ${enabled}`);
    const result = await runPythonScript('simple_recorder.py', ['set-notifications', enabled ? 'True' : 'False']);

    // Extract JSON from output
    const jsonMatch = result.match(/\{.*\}/s);
    if (jsonMatch) {
      const jsonData = JSON.parse(jsonMatch[0]);
      return jsonData;
    }

    return { success: true, notifications_enabled: enabled };
  } catch (error) {
    sendDebugLog(`Error setting notifications: ${error.message}`);
    return { success: false, error: error.message };
  }
});

// Update checking functionality
async function checkForUpdates() {
  return new Promise((resolve) => {
    const options = {
      hostname: 'api.github.com',
      path: '/repos/ruzin/stenoai/releases/latest',
      method: 'GET',
      headers: {
        'User-Agent': 'StenoAI-Updater'
      }
    };

    const req = https.request(options, (res) => {
      let data = '';
      
      res.on('data', (chunk) => {
        data += chunk;
      });
      
      res.on('end', () => {
        try {
          const release = JSON.parse(data);
          const latestVersion = release.tag_name.replace(/^v/, ''); // Remove 'v' prefix if present
          
          // Get current version from package.json
          const packagePath = path.join(__dirname, 'package.json');
          const packageContent = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
          const currentVersion = packageContent.version;
          
          console.log(`Current version: ${currentVersion}, Latest version: ${latestVersion}`);
          
          // Simple version comparison (works for semantic versioning)
          const isUpdateAvailable = compareVersions(currentVersion, latestVersion) < 0;
          
          resolve({
            success: true,
            updateAvailable: isUpdateAvailable,
            currentVersion: currentVersion,
            latestVersion: latestVersion,
            releaseUrl: release.html_url,
            releaseName: release.name || `Version ${latestVersion}`,
            downloadUrl: getDownloadUrl(release.assets)
          });
        } catch (error) {
          console.error('Error parsing GitHub API response:', error);
          resolve({ success: false, error: 'Failed to parse update data' });
        }
      });
    });
    
    req.on('error', (error) => {
      console.error('Error checking for updates:', error);
      resolve({ success: false, error: error.message });
    });
    
    req.setTimeout(10000, () => {
      req.destroy();
      resolve({ success: false, error: 'Update check timeout' });
    });
    
    req.end();
  });
}

function compareVersions(current, latest) {
  const currentParts = current.split('.').map(Number);
  const latestParts = latest.split('.').map(Number);
  
  for (let i = 0; i < Math.max(currentParts.length, latestParts.length); i++) {
    const currentPart = currentParts[i] || 0;
    const latestPart = latestParts[i] || 0;
    
    if (currentPart < latestPart) return -1;
    if (currentPart > latestPart) return 1;
  }
  
  return 0;
}

function getDownloadUrl(assets) {
  // Find the appropriate download URL based on platform/architecture
  const platform = process.platform;
  const arch = process.arch;
  
  if (platform === 'darwin') {
    // Look for macOS DMG files
    const armAsset = assets.find(asset => 
      asset.name.includes('arm64') && asset.name.includes('dmg')
    );
    const intelAsset = assets.find(asset => 
      asset.name.includes('x64') && asset.name.includes('dmg')
    );
    
    // Prefer ARM64 for Apple Silicon, fallback to Intel
    if (arch === 'arm64' && armAsset) return armAsset.browser_download_url;
    if (intelAsset) return intelAsset.browser_download_url;
    if (armAsset) return armAsset.browser_download_url;
  }

  if (platform === 'win32') {
    const archHint = arch === 'arm64' ? 'arm64' : 'x64';
    const winAsset = assets.find(asset =>
      asset.name.toLowerCase().includes('win') &&
      asset.name.toLowerCase().includes(archHint)
    );
    const genericExe = assets.find(asset =>
      asset.name.toLowerCase().endsWith('.exe') || asset.name.toLowerCase().endsWith('.msi')
    );

    if (winAsset) return winAsset.browser_download_url;
    if (genericExe) return genericExe.browser_download_url;
  }
  
  // Fallback to first asset or releases page
  return assets.length > 0 ? assets[0].browser_download_url : null;
}

ipcMain.handle('check-for-updates', async () => {
  return await checkForUpdates();
});

ipcMain.handle('open-release-page', async (event, url) => {
  try {
    await shell.openExternal(url);
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// Real-time transcription handlers
let realtimeTranscriptionProcess = null;

ipcMain.handle('start-realtime-transcription', async (event, options = {}) => {
  try {
    if (realtimeTranscriptionProcess) {
      return { success: false, error: 'Real-time transcription already running' };
    }

    const pythonPath = getVenvPythonPath();
    const scriptPath = path.join(getResourcesRoot(), 'src', 'realtime_transcriber.py');

    sendDebugLog('Starting real-time transcription...');

    realtimeTranscriptionProcess = spawn(pythonPath, [scriptPath], {
      cwd: getResourcesRoot(),
      stdio: ['pipe', 'pipe', 'pipe'],
      env: getPythonEnv()
    });

    realtimeTranscriptionProcess.stdout.on('data', (data) => {
      const output = data.toString();
      // Parse transcript segments and send to frontend
      if (output.includes('[') && output.includes(']:')) {
        mainWindow.webContents.send('realtime-transcript', { text: output.trim() });
      }
      sendDebugLog(output.trim());
    });

    realtimeTranscriptionProcess.stderr.on('data', (data) => {
      sendDebugLog('RT-STDERR: ' + data.toString().trim());
    });

    realtimeTranscriptionProcess.on('close', (code) => {
      sendDebugLog(`Real-time transcription ended with code: ${code}`);
      realtimeTranscriptionProcess = null;
      mainWindow.webContents.send('realtime-transcription-stopped');
    });

    return { success: true, message: 'Real-time transcription started' };
  } catch (error) {
    sendDebugLog(`Real-time transcription error: ${error.message}`);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('stop-realtime-transcription', async () => {
  try {
    if (!realtimeTranscriptionProcess) {
      return { success: false, error: 'No real-time transcription running' };
    }

    realtimeTranscriptionProcess.kill('SIGINT');
    realtimeTranscriptionProcess = null;

    return { success: true, message: 'Real-time transcription stopped' };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('get-realtime-status', async () => {
  return {
    success: true,
    isRunning: realtimeTranscriptionProcess !== null
  };
});
