import * as React from 'react';
import Paper from '@mui/material/Paper';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import { getParameters, deleteParameter } from '../../helpers/api-admin';
import { useEffect } from 'react';
interface Column {
  id: 'name' | 'retriever' | 'generative' | 'database' | 'maxLength' | 'isSelected'| 'deleteID';
  label: string;
  minWidth?: number;
  align?: 'left';
  format?: (value: number) => string;
}

const columns: readonly Column[] = [
  { id: 'name', label: 'Name', minWidth: 100 },
  { id: 'retriever', label: 'Retrieval Model', minWidth: 100 },
  {
    id: 'generative',
    label: 'Generative Model',
    minWidth: 100,
    align: 'left',
  },
  {
    id: 'database',
    label: 'Database Path',
    minWidth: 100,
    align: 'left',
  },
  {
    id: 'maxLength',
    label: 'Max Length',
    minWidth: 100,
    align: 'left',
    format: (value: number) => value.toFixed(2),
  },
  {
    id: 'isSelected',
    label: 'isSelected',
    minWidth: 100,
    align: 'left',
  },
  {
    id: 'deleteID',
    label: 'Delete',
    minWidth: 100,
    align: 'left',
  },
];

interface Data {
  name: string;
  retriever: string;
  generative: string;
  database: string;
  maxLength: number;
  isSelected: boolean;
  deleteID: string;
}

function createData(
  name: string,
  retriever: string,
  generative: string,
  database: string,
  maxLength: number,
  isSelected: boolean,
  deleteID: string
): Data {
  return { name, retriever, generative, database, maxLength, isSelected, deleteID };
}


export default function StickyHeadTableParam() {
  const [page, setPage] = React.useState(0);
  const [rowsPerPage, setRowsPerPage] = React.useState(10);
  const [rows, setRows] = React.useState<Data[]>([]);

  useEffect(() => {
    async function fetchData() {
      const data = await getParameters();
      const updatedRows = data.map((item: { [x: string]: string; }) =>
        createData(
          item['name'],
          item['retriever_model_path_or_name'],
          item['generative_model_path_or_name'],
          item['database_path'],
          parseInt(item['retrieval_max_length']),
          true ? item['isSeletected'] === 'True' : false,
          item['_id']
        )
      );
      setRows(updatedRows);
    }

    fetchData();
  }, []);
  
  const handleDelete = async (id: string) => {
    try {
      await deleteParameter(id);
      const updatedRows = rows.filter((row) => row.deleteID !== id);
      setRows(updatedRows);
    } catch (error) {
      console.error('Error deleting parameter:', error);
    }
  };

  return (
    <Paper sx={{ width: '100%', overflow: 'hidden' }}>
  <TableContainer sx={{ maxHeight: 440 }}>
    <Table stickyHeader aria-label="sticky table" sx={{ background: 'rgba(0, 0, 0, 0.7)', color: 'white' }}>
      <TableHead>
        <TableRow>
          {columns.map((column) => (
            <TableCell
              key={column.id}
              align={column.align}
              style={{
                color: 'white',
                background: 'gray',
                fontWeight: 'bold',
                fontSize: '1.2rem',
                flex: 1, // Set equal flex for each column
                wordWrap: 'break-word', // Enable content wrapping
                overflow: 'hidden', // Hide overflowing text
                textOverflow: 'ellipsis', // Display ellipsis for overflowing text
              }}
            >
              {column.label}
            </TableCell>
          ))}
        </TableRow>
      </TableHead>
      <TableBody>
        {rows
          .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
          .map((row) => {
            return (
              <TableRow hover role="checkbox" tabIndex={-1} key={row.retriever}>
                {columns.map((column) => {
                  const value = row[column.id];
                  return (
                    <TableCell
                      key={column.id}
                      align={column.align}
                      style={{
                        color: 'white',
                        flex: 1, // Set equal flex for each column
                        wordWrap: 'break-word', // Enable content wrapping
                        overflow: 'hidden', // Hide overflowing text
                        textOverflow: 'ellipsis', // Display ellipsis for overflowing text
                      }}
                    >
                      {column.id === 'isSelected' ? (
                        <input type="radio" checked={Boolean(value)} readOnly />
                      ) : column.id === 'deleteID' ? (
                        <button
                          onClick={() => handleDelete(value.toString())}
                          style={{
                            color: 'white',
                            background: 'red',
                            border: 'none',
                            padding: '5px 10px',
                            borderRadius: '5px',
                            cursor: 'pointer',
                          }}
                        >
                          Delete
                        </button>
                      ) : (
                        column.format && typeof value === 'number' ? column.format(value) : value
                      )}
                    </TableCell>
                  );
                })}
              </TableRow>
            );
          })}
      </TableBody>
    </Table>
  </TableContainer>
</Paper>
  );
}